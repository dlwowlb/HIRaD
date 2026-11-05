"""Building blocks of the predictive node-structured latent state."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, logvar, training=True):
    if not training:
        return mu
    return mu + (0.5 * logvar).exp() * torch.randn_like(mu)


def kl_divergence_per_sample(mu_q, lv_q, mu_p, lv_p):
    var_q = lv_q.exp()
    var_p = lv_p.exp().clamp(min=1e-8)
    kl = 0.5 * (lv_p - lv_q + var_q / var_p
                + (mu_q - mu_p).pow(2) / var_p - 1)
    return kl.mean(dim=list(range(1, kl.dim())))


def node_diversity_loss(node_seq, temporal_mask=None):
    """Mean squared off-diagonal cosine similarity between nodes (diagnostic)."""
    B, T, M, D = node_seq.shape
    node_norm = F.normalize(node_seq, dim=-1)
    sim = torch.bmm(
        node_norm.reshape(B * T, M, D),
        node_norm.reshape(B * T, M, D).transpose(1, 2)
    ).reshape(B, T, M, M)
    eye = torch.eye(M, device=sim.device).view(1, 1, M, M)
    off_diag = (sim * (1 - eye)).pow(2)
    if temporal_mask is not None:
        mask = temporal_mask.float().view(B, T, 1, 1)
        return (off_diag * mask).sum() / (mask.sum() * M * (M - 1)).clamp(1)
    return off_diag.sum() / (B * T * M * (M - 1))


class ObservationEncoder(nn.Module):
    """Shared per-point MLP f_obs."""

    def __init__(self, in_dim=4, hidden_dims=None, out_dim=256):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, Y):
        return self.out_norm(self.mlp(Y))


class LatentNodeInitializer(nn.Module):
    """Initializes the node states from the first frame by slot competition."""

    def __init__(self, node_dim, feat_dim, num_heads=4, num_rounds=3):
        super().__init__()
        self.num_rounds = num_rounds
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads,
            kdim=feat_dim, vdim=feat_dim, batch_first=True)
        self.gru = nn.GRUCell(node_dim, node_dim)
        self.norm_slots = nn.LayerNorm(node_dim)
        self.norm_input = nn.LayerNorm(feat_dim)
        self.proj = nn.Linear(node_dim, node_dim)

    def forward(self, node_queries, point_features, point_mask=None):
        B, M, D = node_queries.shape
        if point_mask is not None:
            any_valid = point_mask.any(dim=-1)
            safe_mask = point_mask.clone()
            if (~any_valid).any():
                safe_mask[~any_valid, 0] = True
            key_padding_mask = ~safe_mask
        else:
            key_padding_mask = None

        pf = self.norm_input(point_features)
        slots = node_queries
        for _ in range(self.num_rounds):
            slots_normed = self.norm_slots(slots)
            updates, _ = self.cross_attn(
                slots_normed, pf, pf, key_padding_mask=key_padding_mask)
            slots = self.gru(
                updates.reshape(B * M, D),
                slots.reshape(B * M, D)).reshape(B, M, D)
        return self.proj(slots) + node_queries


class InteractionContextEncoder(nn.Module):
    """Temporal context xi_t from the previous K posterior node states."""

    def __init__(self, node_dim, num_nodes, ctx_dim=128, history_len=5):
        super().__init__()
        self.history_len = history_len
        self.node_dim = node_dim
        self.num_nodes = num_nodes
        self.ctx_dim = ctx_dim
        frame_dim = node_dim * 2 * num_nodes
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_dim, 256), nn.GELU(),
            nn.Linear(256, ctx_dim))
        self.temporal = nn.GRU(
            input_size=ctx_dim, hidden_size=ctx_dim, batch_first=True)
        self.out_norm = nn.LayerNorm(ctx_dim)

    def forward(self, node_history):
        B = node_history[0].shape[0]
        device = node_history[0].device
        K = self.history_len
        padded = list(node_history[-K:])
        zero = torch.zeros(B, self.num_nodes, self.node_dim, device=device)
        while len(padded) < K:
            padded.insert(0, zero)
        frames = []
        for i in range(K):
            nodes = padded[i]
            vel = (nodes - padded[i - 1]) if i > 0 else torch.zeros_like(nodes)
            feat = torch.cat([nodes, vel], dim=-1).reshape(B, -1)
            frames.append(self.frame_proj(feat))
        seq = torch.stack(frames, dim=1)
        _, h_n = self.temporal(seq)
        return self.out_norm(h_n.squeeze(0))


class ContextConditionedTransition(nn.Module):
    """f_trans: inter-node self-attention, FiLM by context, GRU step, prior heads."""

    def __init__(self, node_dim, num_nodes, ctx_dim=128, num_heads=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.self_attn = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.sa_norm = nn.LayerNorm(node_dim)
        self.pre_sa_norm = nn.LayerNorm(node_dim)
        self.ctx_to_gamma_raw = nn.Linear(ctx_dim, num_nodes * node_dim)
        self.ctx_to_beta = nn.Linear(ctx_dim, num_nodes * node_dim)
        self.film_scale = 0.1
        self.gru = nn.GRUCell(node_dim, node_dim)
        self.prior_mu = nn.Linear(node_dim, node_dim)
        self.prior_logvar = nn.Linear(node_dim, node_dim)

    def forward(self, prev_nodes, context):
        B, M, D = prev_nodes.shape
        normed = self.pre_sa_norm(prev_nodes)
        sa_out, _ = self.self_attn(normed, normed, normed)
        nodes = self.sa_norm(prev_nodes + sa_out)
        gamma_raw = self.ctx_to_gamma_raw(context).view(B, M, D)
        beta_raw = self.ctx_to_beta(context).view(B, M, D)
        gamma = 1.0 + self.film_scale * torch.tanh(gamma_raw)
        nodes = gamma * nodes + beta_raw
        prior_nodes = self.gru(
            nodes.reshape(B * M, D),
            prev_nodes.reshape(B * M, D)).reshape(B, M, D)
        mu = self.prior_mu(prior_nodes)
        logvar = self.prior_logvar(prior_nodes).clamp(-6, 2)
        return prior_nodes, mu, logvar


class EvidenceExtractor(nn.Module):
    """Point-to-node routing attention. Returns node evidence and the update gate."""

    def __init__(self, node_dim, feat_dim, num_heads=4,
                 compete_alpha=0.2, num_nodes=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        assert node_dim % num_heads == 0
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(feat_dim, node_dim)
        self.v_proj = nn.Linear(feat_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.scale = self.head_dim ** -0.5
        self.compete_alpha = compete_alpha
        self.node_query_bias = nn.Parameter(
            torch.randn(1, num_nodes, node_dim) * 0.02)
        self.confidence_head = nn.Sequential(
            nn.Linear(num_heads * 2 + 1, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, prior_nodes, point_features, point_mask=None):
        B, M, _ = prior_nodes.shape
        N = point_features.shape[1]
        H, d = self.num_heads, self.head_dim

        Q = self.q_proj(prior_nodes + self.node_query_bias[:, :M, :]).view(B, M, H, d).transpose(1, 2)
        K = self.k_proj(point_features).view(B, N, H, d).transpose(1, 2)
        V = self.v_proj(point_features).view(B, N, H, d).transpose(1, 2)

        logits = (Q @ K.transpose(-2, -1)) * self.scale
        logits = logits.clamp(-30, 30)

        if point_mask is not None:
            vm = point_mask[:, None, None, :].expand_as(logits)
            any_valid = point_mask.any(dim=-1)
            has_invalid = (~any_valid).any()
            logits_masked = logits.masked_fill(~vm, float("-inf"))
            logit_max = logits_masked.max(dim=-1).values
            valid_count = vm.float().sum(dim=-1).clamp(min=1)
            logit_mean = (logits * vm.float()).sum(dim=-1) / valid_count
            n_valid_ratio = point_mask.float().sum(dim=-1) / N
            n_valid = n_valid_ratio.view(B, 1, 1).expand(B, 1, M)
            if has_invalid:
                inv = (~any_valid)[:, None, None].expand_as(logit_max)
                logit_max = logit_max.masked_fill(inv, 0.0)
                logit_mean = logit_mean.masked_fill(inv, 0.0)
            logits = logits.masked_fill(~vm, float("-inf"))
        else:
            any_valid = torch.ones(B, dtype=torch.bool, device=prior_nodes.device)
            has_invalid = False
            logit_max = logits.max(dim=-1).values
            logit_mean = logits.mean(dim=-1)
            n_valid = torch.ones(B, 1, M, device=prior_nodes.device)

        conf_input = torch.cat([
            logit_max.permute(0, 2, 1),
            logit_mean.permute(0, 2, 1),
            n_valid.permute(0, 2, 1),
        ], dim=-1)
        confidence = self.confidence_head(conf_input)

        attn_np = torch.nan_to_num(logits.softmax(dim=-1), nan=0.0)
        if self.compete_alpha > 0:
            attn_pn = torch.nan_to_num(logits.softmax(dim=-2), nan=0.0)
            attn = (1 - self.compete_alpha) * attn_np + self.compete_alpha * attn_pn
        else:
            attn = attn_np

        out = (attn @ V).transpose(1, 2).reshape(B, M, -1)
        evidence = self.out_proj(out)

        if has_invalid:
            mask_f = (~any_valid)[:, None, None].float()
            confidence = confidence * (1 - mask_f)
            evidence = evidence * (1 - mask_f)
        return evidence, confidence


class PosteriorUpdate(nn.Module):
    """f_post: gated fusion of prior and evidence, then inter-node coordination."""

    def __init__(self, node_dim, num_heads=4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim), nn.GELU(),
            nn.Linear(node_dim, node_dim))
        self.fuse_norm = nn.LayerNorm(node_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.sa_norm = nn.LayerNorm(node_dim)
        self.post_mu = nn.Linear(node_dim, node_dim)
        self.post_logvar = nn.Linear(node_dim, node_dim)

    def forward(self, prior, evidence, confidence, training=True):
        fused = self.fuse(torch.cat([prior, evidence], dim=-1))
        gated = confidence * fused + (1 - confidence) * prior
        gated = self.fuse_norm(gated)
        sa_out, _ = self.self_attn(gated, gated, gated)
        coordinated = self.sa_norm(gated + sa_out)
        mu = self.post_mu(coordinated)
        logvar = self.post_logvar(coordinated).clamp(-6, 2)
        posterior = reparameterize(mu, logvar, training=training)
        return posterior, mu, logvar


class NodeWiseDecoder(nn.Module):
    """Reconstructs the observed point cloud from the union of node decoders."""

    def __init__(self, node_dim, num_nodes, num_points=128, point_dim=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_points = num_points
        self.point_dim = point_dim
        self.points_per_node = (num_points + num_nodes - 1) // num_nodes
        self.decoder = nn.Sequential(
            nn.Linear(node_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, self.points_per_node * point_dim))

    def forward(self, nodes):
        B, M, D = nodes.shape
        per_node = self.decoder(nodes)
        per_node = per_node.reshape(B, M * self.points_per_node, self.point_dim)
        return per_node[:, :self.num_points, :]


class NodeTemporalReadout(nn.Module):
    """Temporal attention pooling of each node trajectory: (B, T, M, D) -> (B, M, D)."""

    def __init__(self, node_dim, num_heads=4):
        super().__init__()
        self.temporal_pool = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.temporal_query = nn.Parameter(torch.randn(1, 1, node_dim) * 0.02)
        self.out_norm = nn.LayerNorm(node_dim)

    def forward(self, node_seq, temporal_mask=None):
        B, T, M, D = node_seq.shape
        x = node_seq.permute(0, 2, 1, 3).contiguous().view(B * M, T, D)
        q = self.temporal_query.expand(B * M, -1, -1)
        if temporal_mask is not None:
            kp = ~temporal_mask[:, None, :].expand(B, M, T).reshape(B * M, T)
        else:
            kp = None
        node_emb, _ = self.temporal_pool(q, x, x, key_padding_mask=kp)
        return self.out_norm(node_emb.squeeze(1)).view(B, M, D)


class RadarSemanticHead(nn.Module):
    """Permutation-invariant readout producing the global radar summary g_radar."""

    def __init__(self, node_dim, out_dim=512, num_queries=4,
                 num_heads=4, num_rounds=3, compete_alpha=0.3):
        super().__init__()
        self.num_queries = num_queries
        self.num_rounds = num_rounds
        self.compete_alpha = compete_alpha

        self.token_queries = nn.Parameter(
            self._init_orthogonal(num_queries, node_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.gru = nn.GRUCell(node_dim, node_dim)
        self.norm_q = nn.LayerNorm(node_dim)
        self.norm_k = nn.LayerNorm(node_dim)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(node_dim), nn.Linear(node_dim, out_dim))
        self.global_query = nn.Parameter(torch.randn(1, 1, node_dim) * 0.02)
        self.global_pool = nn.MultiheadAttention(
            embed_dim=node_dim, num_heads=num_heads, batch_first=True)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(node_dim), nn.Linear(node_dim, out_dim))

    @staticmethod
    def _init_orthogonal(num_queries, dim):
        if num_queries <= dim:
            rand = torch.randn(dim, num_queries)
            q, _ = torch.linalg.qr(rand)
            return q[:, :num_queries].t().unsqueeze(0)
        return torch.randn(1, num_queries, dim) * 0.5

    def forward(self, node_emb):
        B, M, D = node_emb.shape
        K = self.num_queries
        kv = self.norm_k(node_emb)
        slots = self.token_queries.expand(B, -1, -1)

        for _ in range(self.num_rounds):
            slots_normed = self.norm_q(slots)
            updates, _ = self.cross_attn(slots_normed, kv, kv)
            slots = self.gru(
                updates.reshape(B * K, D),
                slots.reshape(B * K, D)).reshape(B, K, D)

        q = self.norm_q(slots)
        scores = torch.bmm(q, kv.transpose(1, 2)) / (D ** 0.5)
        attn_qn = scores.softmax(dim=-1)
        attn_nq = scores.softmax(dim=-2)
        attn = (1 - self.compete_alpha) * attn_qn + self.compete_alpha * attn_nq
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sem_tokens = self.token_proj(torch.bmm(attn, kv))

        q_g = self.global_query.expand(B, -1, -1)
        g, _ = self.global_pool(q_g, node_emb, node_emb)
        g = self.global_proj(g.squeeze(1))
        return sem_tokens, g, attn


class NodeAwareReadout(nn.Module):
    """Node summaries n^(m) and global summary g_radar from the node trajectories."""

    def __init__(self, node_dim, out_dim=512,
                 num_semantic_queries=4, num_heads=4):
        super().__init__()
        self.node_temporal = NodeTemporalReadout(node_dim, num_heads)
        self.radar_semantic = RadarSemanticHead(
            node_dim, out_dim, num_semantic_queries, num_heads)

    def forward(self, node_seq, temporal_mask=None):
        node_emb = self.node_temporal(node_seq, temporal_mask)
        sem_tokens, g_radar, attn = self.radar_semantic(node_emb)
        return {
            "node_emb": node_emb,
            "radar_tokens": sem_tokens,
            "g_radar": g_radar,
            "token_attn": attn,
        }
