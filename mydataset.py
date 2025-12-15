import numpy as np
import os
import scipy.io as scio
from torch.utils.data import Dataset

# def normalize(data):                           # 归一化到0-255
#     rawdata_max = max(map(max, data))
#     rawdata_min = min(map(min, data))
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             data[i][j] = round(((255 - 0) * (data[i][j] - rawdata_min) / (rawdata_max - rawdata_min)) + 0)
#     return data

# class MyDataset(Dataset):

#     def __init__(self, root_dir, names_file, transform=None):
#         self.root_dir = root_dir
#         self.names_file = names_file
#         self.transform = transform
#         self.size = 0
#         self.names_list = []
#         if not os.path.isfile(self.names_file):
#             print(self.names_file + 'does not exist!')
#         file = open(self.names_file)
#         for f in file:
#             self.names_list.append(f)
#             self.size += 1

#     def __len__(self):
#         return self.size

#     def __getitem__(self, idx):
#         data_path = self.root_dir + self.names_list[idx].split(' ')[0]
#         if not os.path.isfile(data_path):
#             print(data_path + 'does not exist!')
#             return None
#         rawdata = scio.loadmat(data_path)['data']  # 10000,12 uint16
#         rawdata = rawdata.astype(int)       # int32
#         data = normalize(rawdata)
#         label = int(self.names_list[idx].split(' ')[1])
#         sample = {'data': data, 'label': label}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as scio

# Optionnel: fallback pour .mat v7.3 (HDF5)
try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Standardise the raw Φ‑OTDR matrix by channel using z‑score.

    The original implementation mapped the full range of each sample to the
    interval [0,255] via nested Python loops.  This approach is slow and
    produces large activation magnitudes that can destabilise the network
    during training.  To address the NaN issue observed during training, we
    switch to a vectorised z‑score normalisation:

    .. math:: \tilde{x}_{t,i} = (x_{t,i} - \mu_i) / (\sigma_i + \epsilon)

    where :math:`\mu_i` and :math:`\sigma_i` are the mean and standard
    deviation of channel ``i`` across the temporal dimension ``t``.  A small
    epsilon is added to the denominator to avoid division by zero when the
    channel is constant.  The returned array retains dtype ``np.float32``.

    Args:
        data: raw data matrix of shape [L, S] (e.g. 10000×12) of type
            integer or float.

    Returns:
        Standardised matrix of shape [L, S] with dtype ``np.float32``.
    """
    arr = data.astype(np.float32)
    # Compute per‑channel mean and standard deviation along the temporal axis
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-6  # add epsilon for numerical stability
    # Perform z‑score normalisation
    arr = (arr - mean) / std
    return arr.astype(np.float32)

def load_mat_safely(path: str, key: str = "data") -> np.ndarray:
    """Charge .mat v5 avec scipy, et si besoin v7.3 avec h5py (si dispo)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"Missing or empty: {path}")

    # Essai SciPy (v5)
    try:
        m = scio.loadmat(path)
        if key in m:
            return np.array(m[key])
        # fallback: première ndarray 2D rencontrée
        for k, v in m.items():
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                return np.array(v)
        raise KeyError(f"Key '{key}' not found in {path}")
    except NotImplementedError:
        # v7.3 (HDF5)
        if not _HAS_H5PY:
            raise
        with h5py.File(path, 'r') as hf:
            if key in hf:
                return np.array(hf[key][:])
            ks = list(hf.keys())
            if not ks:
                raise KeyError(f"No datasets in {path}")
            return np.array(hf[ks[0]][:])


class MyDataset(Dataset):
    """
    Attend un fichier 'label' dont chaque ligne: '<rel_path> <label>'
    Exemple: '/01_background/220112_cxm_background_01_single_data_1.mat 0'
    root_dir = dossier 'train' ou 'test' (ex: .../das_data/train)
    Retour: {'data': tensor[L,S], 'label': int}
    """

    def __init__(self, root_dir: str, names_file: str, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform

        if not os.path.isfile(self.names_file):
            raise FileNotFoundError(self.names_file + ' does not exist!')

        # lire sans BOM + nettoyer lignes vides
        with open(self.names_file, "r", encoding="utf-8-sig") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        self.samples = []
        for ln in lines:
            parts = ln.split()
            if len(parts) < 2:
                continue
            rel_path, lab = parts[0], int(parts[1])
            # ⚠️ retire le slash de tête et normalise
            rel_clean = rel_path.lstrip("/\\")
            full_path = os.path.normpath(os.path.join(self.root_dir, rel_clean))
            self.samples.append((full_path, lab))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {self.names_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, lab = self.samples[idx]
        if not os.path.isfile(p):
            raise FileNotFoundError(p + ' does not exist!')

        # charge .mat (clé 'data' par défaut)
        raw = load_mat_safely(p, key="data")   # attendu: (10000,12) uint16
        # corrige orientation si besoin
        if raw.shape == (12, 10000):
            raw = raw.T

        # normalisation 0-255 vectorisée (float32)
        x = normalize(raw)               # shape [L,S] = [10000,12]
        x = torch.from_numpy(x)                # tensor float32

        sample = {'data': x, 'label': lab}
        if self.transform:
            sample = self.transform(sample)
        return sample
