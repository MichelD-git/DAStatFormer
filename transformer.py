# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:55:05 2025

@author: michel
"""

from torch.nn import Module
import torch
from torch.nn import ModuleList
from module.encoder import Encoder #,EncoderV2
import math
import torch.nn.functional as F


class Transformer(Module):
    def __init__(self,
                  d_model: int,
                  d_input: int,
                  d_channel: int,
                  d_output: int,
                  d_hidden: int,
                  q: int,
                  v: int,
                  h: int,
                  N: int,
                  device: str,
                  dropout: float = 0.1,
                  pe: bool = False,
                  mask: bool = False):
        super(Transformer, self).__init__()

        self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.encoder_list_2 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])
        

        self.embedding_channel = torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model)

        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x, stage):
        """
        前向传播
        :param x: 输入
        :param stage: Used to describe whether the process is the training process for the training set or the testing process for the test set.
                      The masking mechanism is not applied during the testing process.
        :return: Output: the two-dimensional vector after the gate, the score matrix in the step-wise encoder, the score matrix in the channel-wise encoder, 
                the three-dimensional matrix after step-wise embedding, the three-dimensional matrix after channel-wise embedding, and the gate
        """
        # step-wise
        # The score matrix is the input, with the default mask and pe added.
        encoding_1 = self.embedding_channel(x)
        input_to_gather = encoding_1

        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_1 = encoding_1 + pe

        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)

        # channel-wise
        # sThe score matrix is channel-based and does not add a mask or pe by default.
        encoding_2 = self.embedding_input(x.transpose(-1, -2))
        channel_to_gather = encoding_2

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage)

        # 3D to 2D
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        # gate
        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

        # output
        output = self.output_linear(encoding)

        return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate

from torch import nn

class Transformer_3d(nn.Module):
    def __init__(self, 
                 d_temp, d_wave, d_spec,      # <--- AJOUT ICI
                 d_model, d_output, d_hidden,
                 q, v, h, N, dropout, pe, mask, device):
        super(Transformer_3d, self).__init__()

        # Embeddings pour chaque groupe de features
        self.embedding_temp = nn.Linear(d_temp, d_model)
        self.embedding_wave = nn.Linear(d_wave, d_model)
        self.embedding_spec = nn.Linear(d_spec, d_model)

        # Encodeurs pour chaque branche
        self.encoder_list_temp = nn.ModuleList([
            Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
                    dropout=dropout, device=device) for _ in range(N)
        ])
        self.encoder_list_wave = nn.ModuleList([
            Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
                    dropout=dropout, device=device) for _ in range(N)
        ])
        self.encoder_list_spec = nn.ModuleList([
            Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
                    dropout=dropout, device=device) for _ in range(N)
        ])

        # Couche de sortie
        self.output_layer = nn.Linear(d_model * 24 * 3, d_output)

        self.pe = pe
        self._d_model = d_model


    def forward(self, x_temp, x_wave, x_spec, stage):
        # print("x_temp.shape:", x_temp.shape)    # (B, 24, 11)
        # print("x_wave.shape:", x_wave.shape)    # (B, 24, 8)
        # print("x_spec.shape:", x_spec.shape)    # (B, 24, 5)
    
        temp_encoded = self.embedding_temp(x_temp)
        wave_encoded = self.embedding_wave(x_wave)
        spec_encoded = self.embedding_spec(x_spec)
    
        # print("temp_encoded.shape:", temp_encoded.shape)  # (B, 24, d_model)
    
        for enc in self.encoder_list_temp:
            temp_encoded, _ = enc(temp_encoded, stage)
        for enc in self.encoder_list_wave:
            wave_encoded, _ = enc(wave_encoded, stage)
        for enc in self.encoder_list_spec:
            spec_encoded, _ = enc(spec_encoded, stage)
    
        temp_encoded = temp_encoded.reshape(temp_encoded.shape[0], -1)
        wave_encoded = wave_encoded.reshape(wave_encoded.shape[0], -1)
        spec_encoded = spec_encoded.reshape(spec_encoded.shape[0], -1)
    
        fused = torch.cat([temp_encoded, wave_encoded, spec_encoded], dim=-1)
        # print("fused.shape:", fused.shape)  # Doit être (B, d_model * 24 * 3)
    
        output = self.output_layer(fused)
        # print("output.shape:", output.shape)  # Doit être (B, d_output)
    
        return output





# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # On suppose que 'Encoder' est disponible et renvoie (encoding, score)
# # from your_module import Encoder

# class Transformer_4d(nn.Module):
#     """
#     Three-branch Transformer with step-wise and channel-wise encoders per branch,
#     gated fusion in-branch, and late fusion across branches.

#     Inputs (forward):
#         x_temp: (B, L=24, d_temp)
#         x_wave: (B, L=24, d_wave)
#         x_spec: (B, L=24, d_spec)

#     Return (tuple):
#         output, fused_all,
#         [score_temp_step, score_wave_step, score_spec_step],
#         [score_temp_chan, score_wave_chan, score_spec_chan],
#         [temp_step_seq,  wave_step_seq,  spec_step_seq],
#         [temp_chan_seq,  wave_chan_seq,  spec_chan_seq],
#         [gate_temp, gate_wave, gate_spec]
#     """
#     def __init__(self,
#                  d_temp: int, d_wave: int, d_spec: int,
#                  d_model: int, d_output: int, d_hidden: int,
#                  q: int, v: int, h: int, N: int,
#                  dropout: float, pe: bool, mask: bool, device: str):
#         super().__init__()
#         self.L = 12
#         self.pe = pe
#         self._d_model = d_model

#         # --- embeddings step-wise (séquence = canaux) ---
#         self.embedding_temp_step = nn.Linear(d_temp, d_model)
#         self.embedding_wave_step = nn.Linear(d_wave, d_model)
#         self.embedding_spec_step = nn.Linear(d_spec, d_model)

#         # --- embeddings channel-wise (séquence = attributs) ---
#         self.embedding_temp_chan = nn.Linear(self.L, d_model)
#         self.embedding_wave_chan = nn.Linear(self.L, d_model)
#         self.embedding_spec_chan = nn.Linear(self.L, d_model)

#         # --- encodeurs step-wise ---
#         self.encoder_list_temp_step = nn.ModuleList([
#             EncoderV2(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
#                     dropout=dropout, device=device) for _ in range(N)
#         ])
#         self.encoder_list_wave_step = nn.ModuleList([
#             EncoderV2(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
#                     dropout=dropout, device=device) for _ in range(N)
#         ])
#         self.encoder_list_spec_step = nn.ModuleList([
#             EncoderV2(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
#                     dropout=dropout, device=device) for _ in range(N)
#         ])

#         # --- encodeurs channel-wise ---
#         self.encoder_list_temp_chan = nn.ModuleList([
#             EncoderV2(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
#                     dropout=dropout, device=device) for _ in range(N)
#         ])
#         self.encoder_list_wave_chan = nn.ModuleList([
#             Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
#                     dropout=dropout, device=device) for _ in range(N)
#         ])
#         self.encoder_list_spec_chan = nn.ModuleList([
#             EncoderV2(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
#                     dropout=dropout, device=device) for _ in range(N)
#         ])

#         # --- gates 2-voies par branche ---
#         self.gate_temp = nn.Linear(d_model * self.L + d_model * d_temp, 2)
#         self.gate_wave = nn.Linear(d_model * self.L + d_model * d_wave, 2)
#         self.gate_spec = nn.Linear(d_model * self.L + d_model * d_spec, 2)

#         # --- tête de sortie ---
#         fused_dim = d_model * (self.L + d_temp) + d_model * (self.L + d_wave) + d_model * (self.L + d_spec)
#         self.output_layer = nn.Linear(fused_dim, d_output)

#     # ---------------- utils ----------------
#     def _add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
#         """Sinusoidal PE"""
#         if not self.pe:
#             return x
#         B, L, D = x.shape
#         device = x.device
#         pe = torch.zeros(L, D, device=device)
#         pos = torch.arange(0, L, dtype=torch.float32, device=device).unsqueeze(1)
#         div = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
#         pe[:, 0::2] = torch.sin(pos * div)
#         pe[:, 1::2] = torch.cos(pos * div)
#         return x + pe.unsqueeze(0)

#     def _branch(self, x, emb_step, emb_chan, encs_step, encs_chan, gate_layer, stage):
#         """
#         x: (B, L, d_attr)
#         Return: fused_branch, gate, step_seq, chan_seq, score_step, score_chan
#         """
#         # step-wise: (B, L, d_attr) -> (B, L, d_model)
#         step_seq = emb_step(x)
#         step_seq = self._add_positional_encoding(step_seq)
#         score_step = None
#         for enc in encs_step:
#             step_seq, score_step = enc(step_seq, stage)  # (B, L, d_model)

#         # channel-wise: (B, d_attr, L) -> (B, d_attr, d_model)
#         chan_seq = emb_chan(x)
#         score_chan = None
#         for enc in encs_chan:
#             chan_seq, score_chan = enc(chan_seq, stage)  # (B, d_attr, d_model)

#         # flatten + gate
#         step_flat = step_seq.reshape(step_seq.size(0), -1)  # (B, L*d_model)
#         chan_flat = chan_seq.reshape(chan_seq.size(0), -1)  # (B, d_attr*d_model)
#         gate = F.softmax(gate_layer(torch.cat([step_flat, chan_flat], dim=-1)), dim=-1)  # (B,2)

#         fused_branch = torch.cat([step_flat * gate[:, 0:1], chan_flat * gate[:, 1:2]], dim=-1)
#         return fused_branch, gate, step_seq, chan_seq, score_step, score_chan

#     # --------------- forward ----------------
#     def forward(self, x_temp, x_wave, x_spec, stage):
#         # branche temporelle
#         x_temp = x_temp.transpose(-1, -2)
#         x_wave = x_wave.transpose(-1, -2)
#         x_spec = x_spec.transpose(-1, -2)
        
#         temp_fused, gate_temp, temp_step_seq, temp_chan_seq, temp_score_step, temp_score_chan = \
#             self._branch(x_temp, self.embedding_temp_step, self.embedding_temp_chan,
#                          self.encoder_list_temp_step, self.encoder_list_temp_chan,
#                          self.gate_temp, stage)

#         # branche waveform
#         wave_fused, gate_wave, wave_step_seq, wave_chan_seq, wave_score_step, wave_score_chan = \
#             self._branch(x_wave, self.embedding_wave_step, self.embedding_wave_chan,
#                          self.encoder_list_wave_step, self.encoder_list_wave_chan,
#                          self.gate_wave, stage)

#         # branche spectrale
#         spec_fused, gate_spec, spec_step_seq, spec_chan_seq, spec_score_step, spec_score_chan = \
#             self._branch(x_spec, self.embedding_spec_step, self.embedding_spec_chan,
#                          self.encoder_list_spec_step, self.encoder_list_spec_chan,
#                          self.gate_spec, stage)

#         # fusion tardive + sortie
#         fused_all = torch.cat([temp_fused, wave_fused, spec_fused], dim=-1)
#         output = self.output_layer(fused_all)

#         # === tuple de sortie ===
#         scores_step_list = [temp_score_step, wave_score_step, spec_score_step]
#         scores_chan_list = [temp_score_chan, wave_score_chan, spec_score_chan]
#         seqs_step_list   = [temp_step_seq,  wave_step_seq,  spec_step_seq]
#         seqs_chan_list   = [temp_chan_seq,  wave_chan_seq,  spec_chan_seq]
#         gates_list       = [gate_temp, gate_wave, gate_spec]

#         return (output, fused_all, scores_step_list, scores_chan_list,
#                 seqs_step_list, seqs_chan_list, gates_list)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assure-toi que ces imports pointent vers tes implémentations
# from .encoder import Encoder, EncoderV2
# Ici on suppose qu'EncoderV2 existe ; si besoin, remplace par Encoder.
EncoderUsed = None
try:
    from module.encoder import EncoderV2 as EncoderUsed
except Exception:
    from module.encoder import Encoder as EncoderUsed


class Transformer_4d(nn.Module):
    """
    3-branch Transformer:
      - Pour chaque branche (temp, wave, spec) :
          * embedding step-wise  : Linear(d_attr -> d_model), séquence = temps (L = d_input)
          * embedding channel-wise: Linear(L -> d_model), séquence = attributs (d_attr)
          * N encodeurs step-wise
          * N encodeurs channel-wise
          * Gate 2-voies (step vs channel), puis concat pondérée
      - Fusion tardive: concat des 3 fusions de branches -> Linear -> logits

    Inputs (forward):
        x_temp: (B, L=d_input, d_temp)
        x_wave: (B, L=d_input, d_wave)
        x_spec: (B, L=d_input, d_spec)
        stage : 'train' | 'test' (retransmis aux encodeurs)

    Returns (tuple):
        logits, fused_all,
        [score_temp_step, score_wave_step, score_spec_step],
        [score_temp_chan, score_wave_chan, score_spec_chan],
        [temp_step_seq,  wave_step_seq,  spec_step_seq],
        [temp_chan_seq,  wave_chan_seq,  spec_chan_seq],
        [gate_temp, gate_wave, gate_spec]
    """

    def __init__(self,
                 d_temp: int, d_wave: int, d_spec: int,
                 d_model: int, d_output: int, d_hidden: int,
                 q: int, v: int, h: int, N: int,
                 dropout: float, pe: bool, mask: bool, device: str,
                 d_input: int = None  # Longueur temporelle (L). Si None, on l'infèrera au 1er forward.
                 ):
        super().__init__()
        self.pe = pe
        self._d_model = d_model
        self._known_L = d_input  # peut être None à l'init, mais requis pour dimensionner les linéaires channel-wise/gates

        # --------- Embeddings step-wise (séquence = temps, longueur L) ----------
        self.embedding_temp_step = nn.Linear(d_temp, d_model)
        self.embedding_wave_step = nn.Linear(d_wave, d_model)
        self.embedding_spec_step = nn.Linear(d_spec, d_model)

        # --------- Embeddings channel-wise (séquence = attributs, longueur = d_attr) ----------
        # Pour ces embeddings, la dimension d'entrée est L (longueur temporelle).
        # Si L n'est pas connu à l'init, on créera ces Linear paresseusement au 1er forward.
        self._needs_lazy_init = (self._known_L is None)
        if not self._needs_lazy_init:
            L = self._known_L
            self.embedding_temp_chan = nn.Linear(L, d_model)
            self.embedding_wave_chan = nn.Linear(L, d_model)
            self.embedding_spec_chan = nn.Linear(L, d_model)
        else:
            self.embedding_temp_chan = None
            self.embedding_wave_chan = None
            self.embedding_spec_chan = None

        # --------- Encodeurs step-wise ----------
        def make_step_stack():
            return nn.ModuleList([
                EncoderUsed(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
                            dropout=dropout, device=device) for _ in range(N)
            ])

        self.encoder_list_temp_step = make_step_stack()
        self.encoder_list_wave_step = make_step_stack()
        self.encoder_list_spec_step = make_step_stack()

        # --------- Encodeurs channel-wise ----------
        def make_chan_stack():
            return nn.ModuleList([
                EncoderUsed(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
                            dropout=dropout, device=device) for _ in range(N)
            ])

        self.encoder_list_temp_chan = make_chan_stack()
        self.encoder_list_wave_chan = make_chan_stack()
        self.encoder_list_spec_chan = make_chan_stack()

        # --------- Gates 2-voies par branche ----------
        # Chaque gate reçoit concat([step_flat (B,L*d_model), chan_flat (B,d_attr*d_model)])
        # => besoin de L et d_attr pour dimensionner. On les instancie paresseusement si L inconnu.
        if not self._needs_lazy_init:
            L = self._known_L
            self.gate_temp = nn.Linear(d_model * (L + d_temp), 2)
            self.gate_wave = nn.Linear(d_model * (L + d_wave), 2)
            self.gate_spec = nn.Linear(d_model * (L + d_spec), 2)
        else:
            self.gate_temp = None
            self.gate_wave = None
            self.gate_spec = None

        # --------- Tête de sortie ----------
        # fused_dim = d_model*(L + d_temp) + d_model*(L + d_wave) + d_model*(L + d_spec)
        if not self._needs_lazy_init:
            L = self._known_L
            fused_dim = d_model * (L + d_temp) + d_model * (L + d_wave) + d_model * (L + d_spec)
            self.output_layer = nn.Linear(fused_dim, d_output)
        else:
            self.output_layer = None  # sera créé au 1er forward

    # ---------------- utils ----------------
    def _lazy_init_if_needed(self, L: int, d_temp: int, d_wave: int, d_spec: int):
        """Instancie les layers dépendant de L au 1er forward si nécessaire."""
        if self.embedding_temp_chan is None:
            self.embedding_temp_chan = nn.Linear(L, self._d_model)
        if self.embedding_wave_chan is None:
            self.embedding_wave_chan = nn.Linear(L, self._d_model)
        if self.embedding_spec_chan is None:
            self.embedding_spec_chan = nn.Linear(L, self._d_model)

        if self.gate_temp is None:
            self.gate_temp = nn.Linear(self._d_model * (L + d_temp), 2)
        if self.gate_wave is None:
            self.gate_wave = nn.Linear(self._d_model * (L + d_wave), 2)
        if self.gate_spec is None:
            self.gate_spec = nn.Linear(self._d_model * (L + d_spec), 2)

        if self.output_layer is None:
            fused_dim = self._d_model * (L + d_temp) + self._d_model * (L + d_wave) + self._d_model * (L + d_spec)
            self.output_layer = nn.Linear(fused_dim, self.d_output)

        self._needs_lazy_init = False
        self._known_L = L

    def _add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Sinusoidal PE sur la dimension séquence (L). x: (B, L, D)"""
        if not self.pe:
            return x
        B, L, D = x.shape
        device = x.device
        pe = torch.zeros(L, D, device=device)
        pos = torch.arange(0, L, dtype=torch.float32, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return x + pe.unsqueeze(0)

    def _branch(self, x, emb_step, emb_chan, encs_step, encs_chan, gate_layer, stage):
        """
        x: (B, L, d_attr)
        Return: fused_branch, gate, step_seq, chan_seq, score_step, score_chan
        """
        # step-wise: (B, L, d_attr) -> (B, L, d_model)
        step_seq = emb_step(x)
        step_seq = self._add_positional_encoding(step_seq)
        score_step = None
        for enc in encs_step:
            step_seq, score_step = enc(step_seq, stage)  # (B, L, d_model)

        # channel-wise: (B, d_attr, L) -> (B, d_attr, d_model)
        chan_seq = emb_chan(x.transpose(-1, -2))
        score_chan = None
        for enc in encs_chan:
            chan_seq, score_chan = enc(chan_seq, stage)  # (B, d_attr, d_model)

        # flatten + gate
        step_flat = step_seq.reshape(step_seq.size(0), -1)   # (B, L*d_model)
        chan_flat = chan_seq.reshape(chan_seq.size(0), -1)   # (B, d_attr*d_model)
        gate = F.softmax(gate_layer(torch.cat([step_flat, chan_flat], dim=-1)), dim=-1)  # (B,2)

        fused_branch = torch.cat([step_flat * gate[:, 0:1], chan_flat * gate[:, 1:2]], dim=-1)
        return fused_branch, gate, step_seq, chan_seq, score_step, score_chan

    # --------------- forward ----------------
    def forward(self, x_temp, x_wave, x_spec, stage: str = 'train'):
        """
        x_temp: (B, L, d_temp) ; x_wave: (B, L, d_wave) ; x_spec: (B, L, d_spec)
        """
        # Dimension L (longueur temps) depuis les entrées
        B, L, d_temp = x_temp.shape
        _, Lw, d_wave = x_wave.shape
        _, Ls, d_spec = x_spec.shape
        assert L == Lw == Ls, f"Incohérence de L: {L} vs {Lw} vs {Ls}"

        # Lazy init si L inconnu à l'init
        if self._needs_lazy_init:
            self.d_output = self.output_layer.out_features if (self.output_layer is not None) else None
            # Si d_output pas connu (cas rare), infère depuis dimensions existantes (non nécessaire ici car passé au ctor)
            self._lazy_init_if_needed(L, d_temp, d_wave, d_spec)

        # Branche temporelle
        temp_fused, gate_temp, temp_step_seq, temp_chan_seq, temp_score_step, temp_score_chan = \
            self._branch(x_temp, self.embedding_temp_step, self.embedding_temp_chan,
                         self.encoder_list_temp_step, self.encoder_list_temp_chan,
                         self.gate_temp, stage)

        # Branche waveform
        wave_fused, gate_wave, wave_step_seq, wave_chan_seq, wave_score_step, wave_score_chan = \
            self._branch(x_wave, self.embedding_wave_step, self.embedding_wave_chan,
                         self.encoder_list_wave_step, self.encoder_list_wave_chan,
                         self.gate_wave, stage)

        # Branche spectrale
        spec_fused, gate_spec, spec_step_seq, spec_chan_seq, spec_score_step, spec_score_chan = \
            self._branch(x_spec, self.embedding_spec_step, self.embedding_spec_chan,
                         self.encoder_list_spec_step, self.encoder_list_spec_chan,
                         self.gate_spec, stage)

        # Fusion tardive + sortie
        fused_all = torch.cat([temp_fused, wave_fused, spec_fused], dim=-1)
        logits = self.output_layer(fused_all)

        # Paquets de sorties (pour logs/visualisation)
        scores_step_list = [temp_score_step, wave_score_step, spec_score_step]
        scores_chan_list = [temp_score_chan, wave_score_chan, spec_score_chan]
        seqs_step_list   = [temp_step_seq,  wave_step_seq,  spec_step_seq]
        seqs_chan_list   = [temp_chan_seq,  wave_chan_seq,  spec_chan_seq]
        gates_list       = [gate_temp, gate_wave, gate_spec]

        return (logits, fused_all,
                scores_step_list, scores_chan_list,
                seqs_step_list,   seqs_chan_list,
                gates_list)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

# On suppose que Encoder et MultiHeadAttention existent déjà dans ton projet,
# exactement comme utilisés par ton modèle GTN 2-branches.

class Transformer_4d1(Module):
    """
    Transformer à 4 branches pour DAS (24 attributs = 11 temporels, 8 waveform, 5 spectraux).

    Branches:
      - temporel-wise  : step-wise sur 11 attributs
      - waveform-wise  : step-wise sur 8 attributs
      - spectral-wise  : step-wise sur 5 attributs
      - channel-wise   : channel-wise sur les 24 attributs (concat des 3 groupes)

    Forward:
      out = model(x_temp, x_wave, x_spec, stage)
      Retourne: (logits, encoding_concat, score_temp, score_wave, score_spec, score_chan,
                 temp_to_gather, wave_to_gather, spec_to_gather, chan_to_gather, gate)
    """
    def __init__(self,
                 d_temp: int,
                 d_wave: int,
                 d_spec: int,
                 d_model: int,
                 d_output: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 device: str,
                 dropout: float = 0.1,
                 pe: bool = False,
                 mask: bool = False):
        super().__init__()

        self.d_temp = d_temp
        self.d_wave = d_wave
        self.d_spec = d_spec
        self.d_model = d_model
        self.d_output = d_output
        self.device = device
        self.pe_flag = pe
        self.mask_flag = mask
        self.N = N

        # --- 3 listes d'encodeurs step-wise (avec mask/PE optionnels) ---
        self.encoder_list_temp = ModuleList([
            Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
                    mask=mask, dropout=dropout, device=device) for _ in range(N)
        ])
        self.encoder_list_wave = ModuleList([
            Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
                    mask=mask, dropout=dropout, device=device) for _ in range(N)
        ])
        self.encoder_list_spec = ModuleList([
            Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
                    mask=mask, dropout=dropout, device=device) for _ in range(N)
        ])

        # --- 1 liste d'encodeurs channel-wise (pas de mask/PE par défaut) ---
        self.encoder_list_chan = ModuleList([
            Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
                    mask=False, dropout=dropout, device=device) for _ in range(N)
        ])

        # --- Embeddings linéaires ---
        # step-wise: on projette chaque "pas de temps" (B, T, d_group) -> (B, T, d_model)
        self.embedding_temp = nn.Linear(d_temp, d_model)
        self.embedding_wave = nn.Linear(d_wave, d_model)
        self.embedding_spec = nn.Linear(d_spec, d_model)

        # channel-wise: on traite les "channels" comme la séquence
        # x_all.transpose(-1, -2) -> (B, C, T), donc in_features = T (d_input)
        # Comme T inconnu à l'init, on utilise un LazyLinear équivalent via init paresseuse.
        self.embedding_input = None  # sera créé au premier forward avec in_features = T

        # --- Gating & sortie ---
        # Dimensions d'entrée (flatten) inconnues avant le premier forward,
        # on crée gate et output_linear au premier passage.
        self.gate = None
        self.output_linear = None

    @staticmethod
    def _build_pe(seq_len: int, d_model: int, device):
        """PE sinusoïdale (T, d_model)."""
        position = torch.arange(0, seq_len, device=device).unsqueeze(-1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(math.log(10000.0) / d_model))  # (d_model/2,)
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe  # (T, d_model)

    def _maybe_build_lazy_linears(self, T: int, C_total: int):
        """
        Initialise paresseusement embedding_input, gate et output_linear
        selon les tailles observées à la première passe.
        """
        if self.embedding_input is None:
            # (B, C, T) -> Linear(T -> d_model) appliqué à chaque channel
            self.embedding_input = nn.Linear(T, self.d_model).to(self.device)

        # tailles flatten
        flat_step = self.d_model * T            # pour chaque branche step-wise
        flat_chan = self.d_model * C_total      # pour la branche channel-wise
        total_in = flat_step * 3 + flat_chan

        if self.gate is None:
            self.gate = nn.Linear(total_in, 4).to(self.device)
        if self.output_linear is None:
            self.output_linear = nn.Linear(total_in, self.d_output).to(self.device)

    def _run_stepwise_branch(self, x_group, encoder_list, add_pe: bool, stage: str):
        """
        x_group: (B, T, d_group) -> embedding -> encoders -> (B, T, d_model)
        Retourne: encoding, score_last, to_gather
        """
        enc = None
        score_last = None

        # embedding
        # (B, T, d_group) -> (B, T, d_model)
        enc = encoder_list[0].MHA  # dummy access to infer device if needed (kept for consistency)
        # Le vrai embedding sera injecté par l'appelant (pour éviter if chain)
        raise_if = False  # placeholder to keep style; not used

        return None, None, None  # will be overridden by caller

    def forward(self, x_temp, x_wave, x_spec, stage: str):
        """
        :param x_temp: (B, T, 11)
        :param x_wave: (B, T, 8)
        :param x_spec: (B, T, 5)
        :param stage : 'train' ou 'test' (le mask n'est pas appliqué en test)
        :return: (logits, encoding_concat, score_temp, score_wave, score_spec, score_chan,
                  temp_to_gather, wave_to_gather, spec_to_gather, chan_to_gather, gate)
        """
        B, T, d_t = x_temp.shape
        _, _, d_w = x_wave.shape
        _, _, d_s = x_spec.shape
        assert d_t == self.d_temp and d_w == self.d_wave and d_s == self.d_spec, \
            f"Bad group sizes: got ({d_t},{d_w},{d_s}), expected ({self.d_temp},{self.d_wave},{self.d_spec})"

        # Concat pour branche channel-wise
        x_all = torch.cat([x_temp, x_wave, x_spec], dim=-1)  # (B, T, 24)
        C_total = x_all.shape[-1]

        # Initialisation paresseuse des couches dépendantes de T / C_total
        self._maybe_build_lazy_linears(T, C_total)

        # ---------- STEP-WISE: Temporel ----------
        enc_temp = self.embedding_temp(x_temp)  # (B, T, d_model)
        temp_to_gather = enc_temp  # avant encodeurs (pour visualisation optionnelle)

        if self.pe_flag:
            pe = self._build_pe(T, self.d_model, enc_temp.device)  # (T, d_model)
            enc_temp = enc_temp + pe.unsqueeze(0)  # broadcast (1, T, d_model)

        score_temp = None
        for enc in self.encoder_list_temp:
            enc_temp, score_temp = enc(enc_temp, stage)  # (B, T, d_model), (B, h, T, T)

        # ---------- STEP-WISE: Waveform ----------
        enc_wave = self.embedding_wave(x_wave)
        wave_to_gather = enc_wave

        if self.pe_flag:
            pe = self._build_pe(T, self.d_model, enc_wave.device)
            enc_wave = enc_wave + pe.unsqueeze(0)

        score_wave = None
        for enc in self.encoder_list_wave:
            enc_wave, score_wave = enc(enc_wave, stage)

        # ---------- STEP-WISE: Spectral ----------
        enc_spec = self.embedding_spec(x_spec)
        spec_to_gather = enc_spec

        if self.pe_flag:
            pe = self._build_pe(T, self.d_model, enc_spec.device)
            enc_spec = enc_spec + pe.unsqueeze(0)

        score_spec = None
        for enc in self.encoder_list_spec:
            enc_spec, score_spec = enc(enc_spec, stage)

        # ---------- CHANNEL-WISE ----------
        # (B, T, C) -> (B, C, T) -> Linear(T -> d_model) -> (B, C, d_model)
        enc_chan = self.embedding_input(x_all.transpose(-1, -2))
        chan_to_gather = enc_chan

        score_chan = None
        for enc in self.encoder_list_chan:
            enc_chan, score_chan = enc(enc_chan, stage)

        # ---------- Flatten + Gate ----------
        flat_temp = enc_temp.reshape(B, -1)  # (B, T*d_model)
        flat_wave = enc_wave.reshape(B, -1)  # (B, T*d_model)
        flat_spec = enc_spec.reshape(B, -1)  # (B, T*d_model)
        flat_chan = enc_chan.reshape(B, -1)  # (B, C*d_model)

        concat_all = torch.cat([flat_temp, flat_wave, flat_spec, flat_chan], dim=-1)  # (B, 3*T*d_model + C*d_model)
        gate_logits = self.gate(concat_all)                                         # (B, 4)
        gate = F.softmax(gate_logits, dim=-1)                                       # (B, 4)

        # Concat pondéré (même logique que GTN 2-branches)
        encoding = torch.cat([
            flat_temp * gate[:, 0:1],
            flat_wave * gate[:, 1:2],
            flat_spec * gate[:, 2:3],
            flat_chan * gate[:, 3:4]
        ], dim=-1)  # (B, total_in)

        # ---------- Sortie ----------
        logits = self.output_linear(encoding)  # (B, d_output)

        return (logits, encoding,
                score_temp, score_wave, score_spec, score_chan,
                temp_to_gather, wave_to_gather, spec_to_gather, chan_to_gather,
                gate)

class Transformer_rd(Module):
    def __init__(self,
                  d_model: int,
                  d_input: int,
                  d_channel: int,
                  d_output: int,
                  d_hidden: int,
                  q: int,
                  v: int,
                  h: int,
                  N: int,
                  device: str,
                  dropout: float = 0.1,
                  pe: bool = False,
                  mask: bool = False):
        super(Transformer_rd, self).__init__()

        self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.encoder_list_2 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.embedding_channel = torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model)

        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x, stage):
        """
        前向传播
        :param x: 输入
        :param stage: Used to describe whether the process is the training process for the training set or the testing process for the test set.
                      The masking mechanism is not applied during the testing process.
        :return: Output: the two-dimensional vector after the gate, the score matrix in the step-wise encoder, the score matrix in the channel-wise encoder, 
                the three-dimensional matrix after step-wise embedding, the three-dimensional matrix after channel-wise embedding, and the gate
        """
        # step-wise
        # The score matrix is the input, with the default mask and pe added.
        x = x.to(torch.float32)
        encoding_1 = self.embedding_channel(x)
        input_to_gather = encoding_1

        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_1 = encoding_1 + pe

        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)

        # channel-wise
        # sThe score matrix is channel-based and does not add a mask or pe by default.
        encoding_2 = self.embedding_input(x.transpose(-1, -2))
        channel_to_gather = encoding_2

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage)

        # 3D to 2D
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        # gate
        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

        # output
        output = self.output_linear(encoding)

        return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate
