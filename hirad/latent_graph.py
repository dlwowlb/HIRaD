"""Predictive node-structured latent state with corrective residuals."""

import gc

import torch
import torch.nn as nn

from .layers import (
    ObservationEncoder, LatentNodeInitializer, InteractionContextEncoder,
    ContextConditionedTransition, EvidenceExtractor, PosteriorUpdate,
    NodeWiseDecoder, NodeAwareReadout, kl_divergence_per_sample,
    node_diversity_loss,
)
from .motion import MotionEncoder, ProjectionHead, info_nce_loss
from .semantic import (
    DeviationDynamicsEncoder, HiddenSemanticHead, QwenTextEncoder,
    semantic_align_loss, confidence_calibration_loss,
)


class LatentGraph(nn.Module):
    """Radar point clouds -> node states, residual trajectories, z_sem.

    At every frame t > 0 the transition predicts a prior (mu_pri, sigma_pri)
    from past posteriors only. The current frame is routed to the nodes to
    form evidence and an update gate, and the posterior (mu_post, sigma_post)
    is produced by the gated update. The corrective residual
    mu_post - mu_pri is stored per node and frame, and the deviation dynamics
    encoder compresses the residual history into the semantic bottleneck.

    `ablation.mode` in the config selects what the semantic pathway reads:
      full                default, residual trajectories
      no_residue_strict   prior trajectories mu_pri (Table 3, variant 4)
      post_mu_input       posterior trajectories mu_post (prediction-subtraction control)
      no_residue          sampled posterior node states
      evidence_input      routed node evidence e_t
    """

    def __init__(self, config):
        super().__init__()
        cfg = config.get("latent_graph", config)

        self.num_nodes = cfg.get("num_nodes", 10)
        self.node_dim = cfg.get("node_dim", 256)
        self.feat_dim = cfg.get("point_feat_dim", 256)
        self.point_in_dim = cfg.get("point_in_dim", 4)
        self.num_points = cfg.get("num_points", 128)
        self.out_dim = cfg.get("out_dim", 512)
        self.ctx_dim = cfg.get("ctx_dim", 128)
        self.ctx_history_len = cfg.get("ctx_history_len", 5)
        num_heads = cfg.get("num_heads", 4)
        self.num_semantic_queries = cfg.get("num_semantic_queries", 4)

        self.beta_kl = cfg.get("beta_kl", 0.1)
        self.contrastive_temp = cfg.get("contrastive_temp", 0.07)
        self.lambda_contrastive = cfg.get("lambda_contrastive", 1.0)
        self.lambda_semantic = cfg.get("lambda_semantic", 0.5)
        self.lambda_conf = cfg.get("lambda_conf", 0.0)
        self.contrastive_dim = cfg.get("contrastive_dim", 256)

        self.node_queries = nn.Parameter(
            self._init_orth(self.num_nodes, self.node_dim))

        self.obs_encoder = ObservationEncoder(
            self.point_in_dim, cfg.get("encoder_hidden", [64, 128]),
            self.feat_dim)
        self.node_init = LatentNodeInitializer(
            self.node_dim, self.feat_dim, num_heads,
            cfg.get("init_rounds", 3))
        self.context_encoder = InteractionContextEncoder(
            self.node_dim, self.num_nodes, self.ctx_dim,
            self.ctx_history_len)
        self.transition = ContextConditionedTransition(
            self.node_dim, self.num_nodes, self.ctx_dim, num_heads)
        self.evidence_extractor = EvidenceExtractor(
            self.node_dim, self.feat_dim, num_heads,
            cfg.get("compete_alpha", 0.2), self.num_nodes)
        self.posterior_update = PosteriorUpdate(self.node_dim, num_heads)
        self.decoder = NodeWiseDecoder(
            self.node_dim, self.num_nodes, self.num_points, 3)
        self.readout = NodeAwareReadout(
            self.node_dim, self.out_dim,
            self.num_semantic_queries, num_heads)

        self.motion_encoder = MotionEncoder(
            num_joints=22, joint_dim=3,
            hidden_dim=cfg.get("motion_hidden", 256),
            out_dim=self.out_dim, num_heads=num_heads)
        self.radar_proj = ProjectionHead(self.out_dim, self.contrastive_dim)
        self.motion_proj = ProjectionHead(self.out_dim, self.contrastive_dim)

        world_cfg = cfg.get("world", {})
        embed_dim = world_cfg.get("embed_dim", 256)
        self.shared_dynamics = DeviationDynamicsEncoder(
            node_dim=self.node_dim,
            global_dim=self.out_dim,
            out_dim=world_cfg.get("shared_dim", 512))
        self.semantic_head = HiddenSemanticHead(
            shared_dim=world_cfg.get("shared_dim", 512),
            embed_dim=embed_dim)

        self._qwen_path = config.get("captioner", {}).get(
            "qwen_path", "Qwen/Qwen3-8B")
        self._embed_dim = embed_dim
        self.text_encoder = None
        self._text_tokenizer = None

        self.ablation_mode = config.get("ablation", {}).get("mode", "full")

    @staticmethod
    def _init_orth(n, d):
        if n <= d:
            q, _ = torch.linalg.qr(torch.randn(d, n))
            return q[:, :n].t().unsqueeze(0)
        return torch.randn(1, n, d) * 0.5

    @staticmethod
    def _point_mask(Y):
        return Y[..., :3].norm(dim=-1) > 1e-6

    def forward_sequence(self, Y, temporal_mask=None):
        B, T, N, _ = Y.shape
        device = Y.device

        if temporal_mask is not None:
            temporal_mask = temporal_mask.clone()
            temporal_mask[:, 0] = True
            for b in range(B):
                vl = temporal_mask[b].long().sum()
                temporal_mask[b, vl:] = False

        U = self.obs_encoder(Y)
        pmask = self._point_mask(Y)
        if temporal_mask is not None:
            temporal_mask = temporal_mask & pmask.any(dim=-1)
            for b in range(B):
                vl = temporal_mask[b].long().sum()
                temporal_mask[b, vl:] = False
            temporal_mask[:, 0] = True

        queries = self.node_queries.expand(B, -1, -1)
        nodes = self.node_init(queries, U[:, 0], pmask[:, 0])

        node_hist = []
        H_node, H_recon, H_conf = [], [], []
        H_deviation, H_prior, H_post_mu, H_evidence = [], [], [], []
        prev_conf = torch.zeros(B, self.num_nodes, 1, device=device)
        kl_sum = kl_w = torch.tensor(0.0, device=device)

        for t in range(T):
            valid_t = temporal_mask[:, t] if temporal_mask is not None \
                else torch.ones(B, dtype=torch.bool, device=device)
            prev_nodes = nodes

            if t == 0:
                ev, conf = self.evidence_extractor(nodes, U[:, t], pmask[:, t])
                new_n, po_mu, _ = self.posterior_update(
                    nodes, ev, conf, self.training)
                deviation = torch.zeros_like(nodes)
                prior_t = nodes
            else:
                c = self.context_encoder(node_hist[-self.ctx_history_len:])
                pr_n, pr_mu, pr_lv = self.transition(nodes, c)
                ev, conf = self.evidence_extractor(pr_n, U[:, t], pmask[:, t])
                new_n, po_mu, po_lv = self.posterior_update(
                    pr_n, ev, conf, self.training)
                kl = kl_divergence_per_sample(po_mu, po_lv, pr_mu, pr_lv)
                kl_sum = kl_sum + (kl * valid_t.float()).sum()
                kl_w = kl_w + valid_t.float().sum()
                deviation = po_mu - pr_mu
                prior_t = pr_mu

            vf = valid_t[:, None, None].float()
            nodes = vf * new_n + (1 - vf) * prev_nodes
            conf = vf * conf + (1 - vf) * prev_conf
            prev_conf = conf.detach()

            node_hist.append(nodes)
            if len(node_hist) > self.ctx_history_len:
                node_hist.pop(0)
            H_node.append(nodes)
            H_recon.append(self.decoder(nodes))
            H_conf.append(conf)
            H_deviation.append(vf * deviation)
            H_prior.append(vf * prior_t)
            H_post_mu.append(vf * po_mu)
            H_evidence.append(vf * ev)

        node_seq = torch.stack(H_node, 1)
        rd = self.readout(node_seq, temporal_mask)
        return {
            "node_history": node_seq,
            "deviation_history": torch.stack(H_deviation, 1),
            "prior_history": torch.stack(H_prior, 1),
            "post_mu_history": torch.stack(H_post_mu, 1),
            "evidence_history": torch.stack(H_evidence, 1),
            "recon_sequence": torch.stack(H_recon, 1),
            "confidence": torch.stack(H_conf, 1),
            "g_radar": rd["g_radar"],
            "radar_tokens": rd["radar_tokens"],
            "node_emb": rd["node_emb"],
            "token_attn": rd["token_attn"],
            "kl": kl_sum / kl_w.clamp(min=1.0),
        }

    def _select_dyn_input(self, out):
        mode = self.ablation_mode
        if mode == "no_residue":
            return out["node_history"]
        if mode == "no_residue_strict":
            return out["prior_history"]
        if mode == "post_mu_input":
            return out["post_mu_history"]
        if mode == "evidence_input":
            return out["evidence_history"]
        return out["deviation_history"]

    def forward(self, point_cloud, motion_joints=None,
                temporal_mask=None, texts=None):
        out = self.forward_sequence(point_cloud, temporal_mask)
        device = point_cloud.device

        loss_obs = self._centered_chamfer(
            out["recon_sequence"], point_cloud[..., :3], temporal_mask)
        loss_kl = out["kl"]
        with torch.no_grad():
            metric_div = node_diversity_loss(out["node_history"], temporal_mask)

        loss_contrastive = torch.tensor(0.0, device=device)
        if motion_joints is not None:
            m_mask = motion_joints[:, :, :22, :3].abs().sum(dim=(-1, -2)) > 1e-6
            motion_emb = self.motion_encoder(
                motion_joints[:, :, :22, :3], mask=m_mask)
            loss_contrastive = info_nce_loss(
                self.radar_proj(out["g_radar"]),
                self.motion_proj(motion_emb),
                self.contrastive_temp)

        s_shared = self.shared_dynamics(
            self._select_dyn_input(out), out["confidence"],
            out["g_radar"], temporal_mask)
        sem = self.semantic_head(s_shared)
        z_sem = sem["z_sem"]
        confidence = sem["semantic_confidence"]

        loss_semantic = torch.tensor(0.0, device=device)
        loss_conf = torch.tensor(0.0, device=device)
        if texts is not None:
            self._ensure_text_encoder(device)
            enc = self._tokenize_texts(texts, device)
            text_emb = self.text_encoder(enc["input_ids"], enc["attention_mask"])
            loss_semantic = semantic_align_loss(
                z_sem, text_emb, self.contrastive_temp)
            loss_conf = confidence_calibration_loss(
                z_sem, text_emb, confidence, self.contrastive_temp)

        loss = (loss_obs
                + self.beta_kl * loss_kl
                + self.lambda_contrastive * loss_contrastive
                + self.lambda_semantic * loss_semantic
                + self.lambda_conf * loss_conf)

        return {
            "loss": loss,
            "loss_obs": loss_obs,
            "loss_kl": loss_kl,
            "loss_contrastive": loss_contrastive,
            "loss_semantic": loss_semantic,
            "loss_conf": loss_conf,
            "metric_div": metric_div,
            "g_radar": out["g_radar"],
            "node_emb": out["node_emb"],
            "node_history": out["node_history"],
            "deviation_history": out["deviation_history"],
            "z_sem": z_sem,
            "semantic_confidence": confidence,
        }

    @staticmethod
    def _centered_chamfer(pred, gt, temporal_mask=None):
        B, T = pred.shape[:2]
        gt_valid = gt.norm(dim=-1) > 1e-6
        total, count = 0.0, 0
        for t in range(T):
            fv = temporal_mask[:, t] if temporal_mask is not None else \
                torch.ones(B, device=pred.device, dtype=torch.bool)
            for b in range(B):
                if not fv[b]:
                    continue
                vm = gt_valid[b, t]
                if vm.sum() < 2:
                    continue
                g, p = gt[b, t, vm], pred[b, t]
                g_n = g - g.mean(0, True)
                p_n = p - p.mean(0, True)
                d = (p_n[:, None] - g_n[None]).pow(2).sum(-1)
                total += d.min(1).values.mean() + d.min(0).values.mean()
                count += 1
        return total / max(count, 1)

    def _ensure_text_encoder(self, device):
        if self.text_encoder is not None:
            return
        from transformers import AutoModelForCausalLM
        lm = AutoModelForCausalLM.from_pretrained(
            self._qwen_path, torch_dtype=torch.float16, device_map="cpu",
            trust_remote_code=True)
        embed_weight = lm.get_input_embeddings().weight.detach().clone()
        del lm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.text_encoder = QwenTextEncoder(embed_weight, self._embed_dim).to(device)

    def _tokenize_texts(self, texts, device):
        if self._text_tokenizer is None:
            from transformers import AutoTokenizer
            self._text_tokenizer = AutoTokenizer.from_pretrained(
                self._qwen_path, trust_remote_code=True)
        enc = self._text_tokenizer(
            texts, padding=True, truncation=True,
            max_length=64, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

    @torch.no_grad()
    def encode(self, point_cloud, temporal_mask=None):
        return self.forward_sequence(point_cloud, temporal_mask)["g_radar"]

    @torch.no_grad()
    def encode_full(self, point_cloud, temporal_mask=None):
        out = self.forward_sequence(point_cloud, temporal_mask)
        s_shared = self.shared_dynamics(
            self._select_dyn_input(out), out["confidence"],
            out["g_radar"], temporal_mask)
        sem = self.semantic_head(s_shared)
        out["z_sem"] = sem["z_sem"]
        out["semantic_confidence"] = sem["semantic_confidence"]
        return out
