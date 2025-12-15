# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:55:05 2025

@author: michel
"""

from torch.nn import Module
import torch
from torch.nn import ModuleList
from module.encoder import Encoder
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



# class Transformer_rd(Module):
#     def __init__(self,
#                   d_model: int,
#                   d_input: int,
#                   d_channel: int,
#                   d_output: int,
#                   d_hidden: int,
#                   q: int,
#                   v: int,
#                   h: int,
#                   N: int,
#                   device: str,
#                   dropout: float = 0.1,
#                   pe: bool = False,
#                   mask: bool = False):
#         super(Transformer_rd, self).__init__()

#         self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
#                                                   d_hidden=d_hidden,
#                                                   q=q,
#                                                   v=v,
#                                                   h=h,
#                                                   mask=mask,
#                                                   dropout=dropout,
#                                                   device=device) for _ in range(N)])

#         self.encoder_list_2 = ModuleList([Encoder(d_model=d_model,
#                                                   d_hidden=d_hidden,
#                                                   q=q,
#                                                   v=v,
#                                                   h=h,
#                                                   dropout=dropout,
#                                                   device=device) for _ in range(N)])

#         self.embedding_channel = torch.nn.Linear(d_channel, d_model)
#         self.embedding_input = torch.nn.Linear(d_input, d_model)

#         self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
#         self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel, d_output)

#         self.pe = pe
#         self._d_input = d_input
#         self._d_model = d_model

#     def forward(self, x, stage):
#         """
#         前向传播
#         :param x: 输入
#         :param stage: Used to describe whether the process is the training process for the training set or the testing process for the test set.
#                       The masking mechanism is not applied during the testing process.
#         :return: Output: the two-dimensional vector after the gate, the score matrix in the step-wise encoder, the score matrix in the channel-wise encoder, 
#                 the three-dimensional matrix after step-wise embedding, the three-dimensional matrix after channel-wise embedding, and the gate
#         """
#         # step-wise
#         # The score matrix is the input, with the default mask and pe added.
#         x = x.to(torch.float32)
#         encoding_1 = self.embedding_channel(x)
#         input_to_gather = encoding_1

#         if self.pe:
#             pe = torch.ones_like(encoding_1[0])
#             position = torch.arange(0, self._d_input).unsqueeze(-1)
#             temp = torch.Tensor(range(0, self._d_model, 2))
#             temp = temp * -(math.log(10000) / self._d_model)
#             temp = torch.exp(temp).unsqueeze(0)
#             temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
#             pe[:, 0::2] = torch.sin(temp)
#             pe[:, 1::2] = torch.cos(temp)

#             encoding_1 = encoding_1 + pe

#         for encoder in self.encoder_list_1:
#             encoding_1, score_input = encoder(encoding_1, stage)

#         # channel-wise
#         # sThe score matrix is channel-based and does not add a mask or pe by default.
#         encoding_2 = self.embedding_input(x.transpose(-1, -2))
#         channel_to_gather = encoding_2

#         for encoder in self.encoder_list_2:
#             encoding_2, score_channel = encoder(encoding_2, stage)

#         # 3D to 2D
#         encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
#         encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

#         # gate
#         gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
#         encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

#         # output
#         output = self.output_linear(encoding)

#         return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate
