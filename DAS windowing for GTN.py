# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 18:53:00 2025

@author: michel
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch.utils.data import Dataset
import scipy.io
import numpy as np
import os

class DASMatDataset(Dataset):
    def __init__(self, root_dir, window_size=200, stride=100, key='data', transform=None):
        self.samples = []
        self.labels = []
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.key = key  # nom de la variable dans le .mat (à adapter selon ton cas)

        class_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for label, class_name in enumerate(class_folders):
            class_path = os.path.join(root_dir, class_name)
            for file in os.listdir(class_path):
                if file.endswith('.mat'):
                    full_path = os.path.join(class_path, file)
                    mat = scipy.io.loadmat(full_path)
                    if self.key not in mat:
                        raise ValueError(f"Clé '{self.key}' introuvable dans le fichier {file}")
                    data = mat[self.key]  # <- doit être shape (10000, 12)
                    if data.shape[0] < window_size:
                        continue  # ignorer si trop court
                    for i in range(0, data.shape[0] - window_size + 1, stride):
                        window = data[i:i+window_size]
                        self.samples.append(window)
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx].astype(np.float32)
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

from torch.utils.data import DataLoader
import torch

# === Dataset personnalisé (défini auparavant) ===
# from ton_module import DASMatDataset

dataset = DASMatDataset(
    root_dir=r"C:\Users\michel\Downloads\Das_data\train",
    window_size=200,
    stride=100,
    key="data"
)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# === Boucle sur tout le dataset ===
total = 0
for i, (x, y) in enumerate(train_loader):
    print(f"Batch {i} → x: {x.shape}, y: {y.shape}")
    total += x.shape[0]

print(f"\nNombre total d'échantillons (fenêtres): {len(dataset)}")
print(f"Nombre total vu dans DataLoader : {total}")
