"""Language-aligned semantic bottleneck: residual histories -> z_sem."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QwenTextEncoder(nn.Module):
    """Caption embedding from the frozen input embeddings of the language model.

    Masked mean pooling over token embeddings followed by a small trainable
    projection into the bottleneck space.
    """

    def __init__(self, qwen_embed_weight, embed_dim=256):
        super().__init__()
        qwen_dim = qwen_embed_weight.shape[1]
        self.word_embed = nn.Embedding.from_pretrained(
            qwen_embed_weight, freeze=True)
        self.proj = nn.Sequential(
            nn.Linear(qwen_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim))

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            emb = self.word_embed(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(1)
        else:
            pooled = emb.mean(dim=1)
        return F.normalize(self.proj(pooled), dim=-1)


class DeviationDynamicsEncoder(nn.Module):
    """Deviation dynamics encoder.

    Per-node temporal convolution over the residual history, node attention
    conditioned on the residual code together with the mean gate and mean
    residual norm, and fusion with the global radar summary.
    """

    def __init__(self, node_dim=256, global_dim=512, out_dim=512):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(node_dim, node_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(node_dim, node_dim, kernel_size=5, padding=2),
            nn.GELU())
        self.dev_magnitude_proj = nn.Sequential(
            nn.Linear(1, node_dim // 4),
            nn.GELU(),
            nn.Linear(node_dim // 4, node_dim))
        self.node_attn = nn.Sequential(
            nn.Linear(node_dim + 2, node_dim),
            nn.Tanh(),
            nn.Linear(node_dim, 1))
        self.fuse = nn.Sequential(
            nn.LayerNorm(node_dim + global_dim),
            nn.Linear(node_dim + global_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim))

    def forward(self, deviation_history, confidence, g_radar,
                temporal_mask=None):
        """
        deviation_history: (B, T, M, D) residuals mu_post - mu_pri
        confidence:        (B, T, M, 1) update gates
        g_radar:           (B, global_dim)
        temporal_mask:     (B, T) valid frames
        """
        B, T, M, D = deviation_history.shape

        x = deviation_history.permute(0, 2, 3, 1).reshape(B * M, D, T)
        x = self.temporal_conv(x)
        if temporal_mask is not None:
            mask = temporal_mask.unsqueeze(1).expand(-1, M, -1)
            mask = mask.reshape(B * M, T).unsqueeze(1).float()
            x = (x * mask).sum(-1) / mask.sum(-1).clamp(1)
        else:
            x = x.mean(-1)
        x = x.view(B, M, D)

        dev_norm = deviation_history.norm(dim=-1, keepdim=True)
        if temporal_mask is not None:
            t_mask = temporal_mask.unsqueeze(-1).unsqueeze(-1).float()
            dev_mean = (dev_norm * t_mask).sum(dim=1) / t_mask.sum(dim=1).clamp(1)
            conf_mean = (confidence * t_mask).sum(dim=1) / t_mask.sum(dim=1).clamp(1)
        else:
            dev_mean = dev_norm.mean(dim=1)
            conf_mean = confidence.mean(dim=1)

        node_features = torch.cat([x, conf_mean, dev_mean], dim=-1)
        attn_weights = F.softmax(self.node_attn(node_features), dim=1)
        dynamics = (x * attn_weights).sum(dim=1)
        return self.fuse(torch.cat([dynamics, g_radar], dim=-1))


class HiddenSemanticHead(nn.Module):
    """s_shared -> l2-normalized semantic bottleneck z_sem."""

    def __init__(self, shared_dim=512, embed_dim=256, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.sem_proj = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.GELU(),
            nn.Linear(shared_dim // 2, embed_dim))
        self.conf_head = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid())

    def forward(self, s_shared):
        z_raw = self.sem_proj(s_shared)
        semantic_confidence = torch.sigmoid(z_raw.norm(dim=-1).log())
        return {
            "z_sem": F.normalize(z_raw, dim=-1),
            "z_raw": z_raw,
            "semantic_confidence": semantic_confidence,
        }


def semantic_align_loss(z_sem, text_emb, temperature=0.07):
    """Symmetric InfoNCE between z_sem and caption embeddings (L_sem)."""
    logits = z_sem @ text_emb.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.t(), labels)) / 2


def confidence_calibration_loss(z_sem, text_emb, confidence,
                                temperature=0.07, scale=5.0):
    """Optional calibration term. Disabled in the released config (weight 0)."""
    B = z_sem.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z_sem.device)
    with torch.no_grad():
        sims = z_sem @ text_emb.t() / temperature
        self_sim = sims.diag()
        eye = torch.eye(B, device=sims.device).bool()
        other_max = sims.masked_fill(eye, -float("inf")).max(dim=1).values
        target_conf = torch.sigmoid((self_sim - other_max) * scale)
    return F.binary_cross_entropy(confidence, target_conf)
