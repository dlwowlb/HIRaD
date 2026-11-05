"""Paired body-motion branch used only as auxiliary supervision (L_ctr)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEncoder(nn.Module):
    """Joint positions (B, T, 22, 3) -> motion embedding (B, out_dim)."""

    def __init__(self, num_joints=22, joint_dim=3, hidden_dim=256,
                 out_dim=512, num_heads=4):
        super().__init__()
        in_dim = num_joints * joint_dim
        self.spatial = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim))
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU())
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim))

    def forward(self, joints, mask=None):
        B, T, J, D = joints.shape
        x = self.spatial(joints.reshape(B, T, J * D))
        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        q = self.pool_query.expand(B, -1, -1)
        kp = ~mask if mask is not None else None
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=kp)
        return self.out_proj(pooled.squeeze(1))


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim))
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        return self.norm(self.net(x))


def info_nce_loss(emb_a, emb_b, temperature=0.07):
    a = F.normalize(emb_a, dim=-1)
    b = F.normalize(emb_b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.t(), labels)) / 2
