# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:44:45 2025

@author: michel
"""

from torch.nn import Module
import torch
from torch.nn import CrossEntropyLoss

class Myloss(Module):
    def __init__(self):
        super(Myloss, self).__init__()
        self.loss_function = CrossEntropyLoss()

    def forward(self, y_pre, y_true):
        y_true = y_true.long()
        loss = self.loss_function(y_pre, y_true)

        return loss