# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:34:53 2025

@author: michel
"""

from torch.nn import Module
import torch

from module.feedForward import FeedForward
from module.multiHeadAttention import MultiHeadAttention

class Encoder(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()

        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, device=device, dropout=dropout)
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, stage):

        residual = x
        x, score = self.MHA(x, stage)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Utilitaires ----
class DropPath(nn.Module):
    """Stochastic depth (per sample)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask

class LayerScale(nn.Module):
    """Résidu pondéré par un gain appris (initialisé petit)."""
    def __init__(self, d_model: int, init_scale: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_scale * torch.ones(d_model))
    def forward(self, x):
        return x * self.gamma

class RelPosBias(nn.Module):
    """
    Biais positionnel relatif apprenable (style T5).
    Utilise des distances clipping à max_distance.
    """
    def __init__(self, num_heads: int, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bias = nn.Parameter(torch.zeros(num_heads, 2*max_distance - 1))
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, qlen: int, klen: int):
        device = self.bias.device
        # positions relatives: i-j dans [-max+1, max-1]
        ctx = torch.arange(qlen, device=device)[:, None] - torch.arange(klen, device=device)[None, :]
        ctx = ctx.clamp(-self.max_distance+1, self.max_distance-1)
        ctx = ctx + (self.max_distance - 1)  # shift to [0 .. 2*max-2]
        # shape: (num_heads, qlen, klen)
        rel = self.bias[:, ctx]  # indexation broadcastée
        return rel  # (H, Q, K)

# ---- Multi-Head Attention minimal, avec biais relatif optionnel ----
class MultiHeadAttentionV2(nn.Module):
    def __init__(self, d_model, num_heads, attn_dropout=0.1, proj_dropout=0.1,
                 use_rel_pos=False, max_rel_pos=128, device="cpu"):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3*d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.use_rel_pos = use_rel_pos
        self.rel_bias = RelPosBias(num_heads, max_distance=max_rel_pos) if use_rel_pos else None

    def forward(self, x, mask=None, return_attn=True):
        # x: (B, L, d_model)
        B, L, D = x.shape
        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape en têtes
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,Hd)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,L,L)

        if self.use_rel_pos:
            # ajout biais relatif par tête, broadcast sur B
            rel = self.rel_bias(L, L)  # (H,L,L)
            attn = attn + rel.unsqueeze(0)

        if mask is not None:
            # mask: (B,1,1,L) ou (B,1,L,L); adapter à ton usage
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # (B,H,L,Hd)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out(out)
        out = self.proj_drop(out)

        return out, (attn if return_attn else None)

# ---- FFN avec GEGLU/SwiGLU ----
class FeedForwardV2(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1, ffn_type="geglu"):
        super().__init__()
        self.ffn_type = ffn_type.lower()
        if self.ffn_type in ["geglu", "swiglu"]:
            self.proj_in = nn.Linear(d_model, 2*d_hidden)
            self.proj_out = nn.Linear(d_hidden, d_model)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_model)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.ffn_type == "geglu":
            a, b = self.proj_in(x).chunk(2, dim=-1)
            x = F.gelu(b) * a
            x = self.proj_out(x)
            return self.dropout(x)
        elif self.ffn_type == "swiglu":
            a, b = self.proj_in(x).chunk(2, dim=-1)
            x = F.silu(b) * a
            x = self.proj_out(x)
            return self.dropout(x)
        else:
            return self.net(x)

# ---- Encoder V2 ----
class EncoderV2(nn.Module):
    """
    Encoder plus robuste pour Transformer_3d.
    - PreNorm (LN avant MHA/FFN)
    - RelPosBias optionnel (use_rel_pos=True) pour L variable (24 canaux ou d_attr)
    - FFN GEGLU/SwiGLU
    - LayerScale + DropPath (stochastic depth) optionnels
    """
    def __init__(self,
                  d_model: int,
                  d_hidden: int,
                  q: int, v: int, h: int,       # conservés pour compat
                  device: str,
                  mask: bool = False,           # compat param
                  dropout: float = 0.1,
                  attn_dropout: float = 0.1,
                  ffn_dropout: float = 0.1,
                  drop_path: float = 0.0,       # stochastic depth
                  layerscale_init: float = 1e-4,
                  use_rel_pos: bool = True,
                  max_rel_pos: int = 128,
                  ffn_type: str = "geglu",
                  return_attn: bool = True):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionV2(
            d_model=d_model, num_heads=h,
            attn_dropout=attn_dropout, proj_dropout=dropout,
            use_rel_pos=use_rel_pos, max_rel_pos=max_rel_pos, device=device
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.layerscale1 = LayerScale(d_model, init_scale=layerscale_init)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn  = FeedForwardV2(d_model=d_model, d_hidden=d_hidden,
                                  dropout=ffn_dropout, ffn_type=ffn_type)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.layerscale2 = LayerScale(d_model, init_scale=layerscale_init)

        self.return_attn = return_attn

    def forward(self, x, stage=None, mask=None):
        # x: (B, L, d_model)
        # --- PreNorm + MHA ---
        y, score = self.attn(self.norm1(x), mask=mask, return_attn=self.return_attn)
        x = x + self.drop_path1(self.layerscale1(y))

        # --- PreNorm + FFN ---
        y = self.ffn(self.norm2(x))
        x = x + self.drop_path2(self.layerscale2(y))

        return x, score
