# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 10:46:42 2025

@author: michel
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

#coding = UTF-8


from scipy.io import loadmat



import datetime
from sklearn import svm, preprocessing
# from get_das_data import get_das_data,get_stats_features
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import seaborn as sns

# from transformer import ViT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# data_train = pd.read_csv(r"C:\Users\michel\Downloads\Phi-OTDR_dataset_and_codes-main\Temporal_freq_feature_data_with_more_features_fft.csv", header=None)
# data_test = pd.read_csv(r"C:\Users\michel\Downloads\Phi-OTDR_dataset_and_codes-main\5km_10km_svm_feature_data_with_more_features_fft.csv", header=None)

# # path = r"C:\Users\michel\Downloads\Phi-OTDR_dataset_and_codes-main"

# X_train = data_train.iloc[:, :-1].values
# y_train = data_train.iloc[:, -1].values
# X_test = data_test.iloc[:, :-1].values
# y_test = data_test.iloc[:, -1].values

# # Normaliser correctement chaque feature
# # minMaxScaler = preprocessing.MinMaxScaler()
# # X_train = minMaxScaler.fit_transform(X_train)
# # X_test = minMaxScaler.transform(X_test)

# scaler = StandardScaler().fit(X_train)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# # print(f"Shape total: {X.shape}, Shape train: {X_train.shape}, Shape test: {X_test.shape}")

# # Normaliser correctement chaque feature
# # minMaxScaler = preprocessing.MinMaxScaler()
# # X_train = minMaxScaler.fit_transform(X_train)
# # X_test = minMaxScaler.transform(X_test)

# def normalize(data):
#     data = np.array(data)
#     data = (data - data.min()) / (data.max() - data.min()) * 255
#     return data

# Xtrain = X_train.reshape(-1, 24, 24)
# Xtest = X_test.reshape(-1, 24, 24)


# from scipy.io import savemat

# def structure_transformer(X,y): 
    
#     x_object = np.empty((1, X.shape[0]), dtype=object)
    
#     for i in range(y.shape[0]):
#         x_object[0, i] = X[i]
#     y_object = y.reshape(y.shape[0],1)
        
#     return x_object, y_object


# Xtrain_gtn, ytrain_gtn = structure_transformer(Xtrain,y_train)

# Xtest_gtn, ytest_gtn = structure_transformer(Xtest,y_test)



# # === 6. Créer une structure de type MATLAB ===
# gtn_struct = np.array([(ytrain_gtn, Xtrain_gtn, ytest_gtn, Xtest_gtn)],
#                       dtype=[('trainlabels', 'O'),
#                              ('train', 'O'),
#                              ('testlabels', 'O'),
#                              ('test', 'O')])

# # === 7. Sauvegarder au format .mat ===
# savemat("GTN_dataset_structured.mat", {"GTN_dataset": gtn_struct})

# print("y_train min:", y_train.min(), "max:", y_train.max())
# print("y_test min:", y_test.min(), "max:", y_test.max())



from torch.utils.data import DataLoader
from dataset_process.dataset_process import MyDataset
import torch.optim as optim
from time import time
from tqdm import tqdm
import os

from transformer import Transformer
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization

class Transformer1D(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
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
        super(Transformer1D, self).__init__()

        self.encoder_list = ModuleList([
            Encoder(d_model=d_model,
                    d_hidden=d_hidden,
                    q=q,
                    v=v,
                    h=h,
                    mask=mask,
                    dropout=dropout,
                    device=device)
            for _ in range(N)
        ])

        self.embedding = torch.nn.Linear(1, d_model)  # pour des signaux 1D (valeur scalaire à chaque pas de temps)
        self.output_linear = torch.nn.Linear(d_model * d_input, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x, stage):
        """
        x: (batch, length) —> signal 1D
        """
        x = x.unsqueeze(-1)  # (batch, length, 1)
        x = self.embedding(x)  # (batch, length, d_model)

        if self.pe:
            pe = torch.ones_like(x[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)
            x = x + pe

        for encoder in self.encoder_list:
            x, score = encoder(x, stage)

        x = x.reshape(x.shape[0], -1)  # (batch, d_input * d_model)
        output = self.output_linear(x)  # (batch, d_output)
        return output
