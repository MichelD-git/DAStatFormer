import scipy.io as scio
import numpy as np
from feature_extraction import feature_extraction,feature_extraction_selected24

def get_diff_data(data):                       # 数据差分
    m = data.shape[0]        # 10000
    n = data.shape[1]        # 12
    data_diff = np.empty([m-1, n])   # 9999,12
    for i in range(m-1):
        data_diff[i, :] = data[i+1, :] - data[i, :]
    return data_diff

def get_feature_list(data):                     # 10000,12
    sample_feature_list = np.zeros([1700,24])    # 提特征 12,24
    for i in range(12):
        f_data = data[:, i]                         # 10000,1 / 9999,1
        feature_list = feature_extraction_selected24(f_data)   # 24
        # print("rgtghrh",len(feature_list))
        sample_feature_list[i, :] = feature_list
        # print(sample_feature_list.shape)
    return sample_feature_list

def get_das_data(rootpath, labelpath):
    datapath = rootpath
    file = open(labelpath)
    name_list = []
    for f in file:
        name_list.append(f)
    temp = np.empty([len(name_list), 1700, 24])
    label_temp = np.empty(len(name_list))
    for i in range(len(name_list)):
        path = datapath + name_list[i].split(' ')[0]
        rawdata = scio.loadmat(path)['data']              # 原始数据 10000,12
        # diffdata = get_diff_data(rawdata)               # 差分数据 9999,12
        rawdata_sample_feature_list = get_feature_list(rawdata)   # 原始数据特征 12,24
        # diffdata_sample_feature_list = get_feature_list(diffdata)  # 差分数据特征 12,24
        sample_feature_list = rawdata_sample_feature_list #np.concatenate((rawdata_sample_feature_list, diffdata_sample_feature_list), axis=1)   # 两类特征合并 12,48
        # print(sample_feature_list.shape)
        feature_data = np.reshape(sample_feature_list, (1, sample_feature_list.shape[0], sample_feature_list.shape[1]))
        temp[i, :, :] = feature_data
        label = int(name_list[i].split(' ')[1])
        label_temp[i] = label
    temp = temp.reshape(len(name_list), -1)  # 样本量，12*48（展平）
    return temp, label_temp




#### Get Da data for DAS data Luna

# das_dataset_npz.py
"""
Utilities to load a DAS dataset organized as:
root_dir/
  car/
    *.npz
  no_car/
    *.npz

Each .npz is expected to contain one 2D array shaped (T, C):
- T = number of time samples (e.g., 10000)
- C = number of channels (e.g., 12)

We keep using your existing `feature_extraction(f_data)` that returns a 1D
vector of length 24 (statistics) for a single 1D signal.
For each file:
- Compute features on raw data (per channel): shape (C, 24)
- Compute features on first-order differences over time (per channel): shape (C, 24)
- Concatenate along feature axis -> (C, 48)
- Optionally flatten to 1D -> (C*48,)

Returns:
- X: np.ndarray of shape (N, C*48) by default (flatten=True) or (N, C, 48)
- y: np.ndarray of shape (N,), where car -> 1, no_car -> 0 (configurable)

Robust loading:
- If the .npz has multiple arrays, the first 2D array encountered is used,
  or a key named 'data' (case-insensitive) if present.
# """

# from __future__ import annotations
# import numpy as np
# from pathlib import Path
# from typing import Tuple, List, Optional, Dict, Any, Sequence

# # Import your feature extractor
# from feature_extraction import feature_extraction


# def _load_npz_main_array(npz_path: Path) -> np.ndarray:
#     """Load the main 2D array (T, C) from an .npz file.
#     Priority:
#       1) key named 'data' (case-insensitive) that is 2D
#       2) the first 2D array encountered
#     Raises ValueError if none found.
#     """
#     with np.load(npz_path, allow_pickle=False) as npz:
#         # Try 'data' (any case)
#         for k in npz.files:
#             if k.lower() == 'data':
#                 arr = npz[k]
#                 if arr.ndim == 2:
#                     return arr
#         # Else first 2D
#         for k in npz.files:
#             arr = npz[k]
#             if arr.ndim == 2:
#                 return arr
#     raise ValueError(f"No 2D array found in {npz_path.name}. Keys={list(np.load(npz_path).files)}")


# def get_diff_data(data: np.ndarray) -> np.ndarray:
#     """First-order difference over time axis (axis=0).
#     Input:  data shape (T, C)
#     Output: diff shape (T-1, C)
#     """
#     if data.ndim != 2:
#         raise ValueError(f"Expected 2D array (T, C), got shape {data.shape}")
#     return np.diff(data, axis=0)


# def get_feature_list(data: np.ndarray) -> np.ndarray:
#     """Compute per-channel features.
#     Input:  data shape (T, C)
#     Output: features shape (C, 24)
#     """
#     if data.ndim != 2:
#         raise ValueError(f"Expected 2D array (T, C), got shape {data.shape}")
#     T, C = data.shape
#     feats = np.zeros((C, 24), dtype=float)
#     for ch in range(C):
#         f_data = data[:, ch]
#         feats[ch, :] = feature_extraction(f_data)
#     return feats


# def extract_features_from_array(rawdata: np.ndarray) -> np.ndarray:
#     """From raw (T, C) produce per-channel concatenated features (C, 48)."""
#     diffdata = get_diff_data(rawdata)                       # (T-1, C)
#     raw_feats = get_feature_list(rawdata)                   # (C, 24)
#     diff_feats = get_feature_list(diffdata)                 # (C, 24)
#     sample_feats = np.concatenate([raw_feats, diff_feats], axis=1)  # (C, 48)
#     return sample_feats


# def _iter_files(folder: Path, suffixes: Sequence[str]=('.npz',)) -> List[Path]:
#     files = []
#     for suf in suffixes:
#         files.extend(sorted(folder.glob(f'*{suf}')))
#     return files


# def load_das_data_from_folders(
#     root_dir: Path | str,
#     labels: Sequence[str] = ('no_car', 'car'),
#     label_map: Optional[Dict[str, int]] = None,
#     flatten: bool = True,
#     verbose: bool = True,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Load dataset from subfolders under root_dir.
    
#     Args:
#         root_dir: path to the dataset root containing subfolders per class.
#         labels: names of the subfolders to scan, order defines label indices
#                 unless label_map is provided. Default ('no_car', 'car').
#         label_map: optional explicit mapping folder_name -> int label.
#                    If None, uses {labels[i]: i for i in range(len(labels))}.
#         flatten: if True returns X with shape (N, C*48); else (N, C, 48).
#         verbose: print progress / warnings.
        
#     Returns:
#         X: np.ndarray, shape (N, C*48) if flatten else (N, C, 48)
#         y: np.ndarray, shape (N,), int labels
#     """
#     root = Path(root_dir)
#     if label_map is None:
#         label_map = {name: idx for idx, name in enumerate(labels)}
#     else:
#         # Ensure all listed labels exist in mapping
#         for name in labels:
#             if name not in label_map:
#                 raise ValueError(f"Label '{name}' not present in provided label_map={label_map}")
    
#     X_list: List[np.ndarray] = []
#     y_list: List[int] = []
#     n_channels_ref: Optional[int] = None
    
#     for label_name in labels:
#         class_dir = root / label_name
#         if not class_dir.is_dir():
#             if verbose:
#                 print(f"[WARN] Missing folder: {class_dir}. Skipping.")
#             continue
        
#         files = _iter_files(class_dir, suffixes=('.npz',))
#         if verbose:
#             print(f"[INFO] {label_name}: {len(files)} files")
        
#         for fpath in files:
#             try:
#                 raw = _load_npz_main_array(fpath)  # (T, C)
#                 # Consistency check for channels
#                 if n_channels_ref is None:
#                     n_channels_ref = raw.shape[1]
#                 elif raw.shape[1] != n_channels_ref:
#                     if verbose:
#                         print(f"[WARN] {fpath.name}: channel mismatch {raw.shape[1]} != {n_channels_ref}. Skipping.")
#                     continue
                
#                 feats = extract_features_from_array(raw)  # (C, 48)
#                 X_list.append(feats if not flatten else feats.reshape(-1))
#                 y_list.append(label_map[label_name])
            
#             except Exception as e:
#                 if verbose:
#                     print(f"[ERROR] {fpath.name}: {e}. Skipped.")
#                 continue
    
#     if not X_list:
#         raise RuntimeError(f"No samples loaded from {root}. Check folder names and .npz contents.")
    
#     X = np.stack(X_list, axis=0)  # (N, C, 48) or (N, C*48)
#     y = np.asarray(y_list, dtype=int)
#     return X, y


# def demo_usage():
#     """Example usage.
#     Adjust `data_root` to your dataset path.
#     """
#     data_root = Path("path/to/root_dir")  # contains 'car' and 'no_car'
#     # By default: ('no_car', 'car') -> labels 0 and 1 respectively
#     X, y = load_das_data_from_folders(data_root, labels=('no_car', 'car'), flatten=True)
#     print("X shape:", X.shape)  # (N, C*48)
#     print("y shape:", y.shape)  # (N,)
#     print("Class balance:", {cls: int((y==cls).sum()) for cls in np.unique(y)})




# def get_das_data_separate_domains(rootpath, labelpath):
#     datapath = rootpath
#     file = open(labelpath)
#     # print(file)
#     name_list = []
#     for f in file:
#         name_list.append(f)
    
#     Time_temp = np.empty([len(name_list), 12, 11])
#     Waveform_temp = np.empty([len(name_list), 12, 8])
#     Spectral_temp = np.empty([len(name_list), 12, 5])
#     temp = np.empty([len(name_list), 12, 24])
#     label_temp = np.empty(len(name_list))
#     for i in range(len(name_list)):
#         path = datapath + name_list[i].split(' ')[0]
#         rawdata = scio.loadmat(path)['data']              # 原始数据 10000,12
#         # diffdata = get_diff_data(rawdata)               # 差分数据 9999,12
#         Time_rawdata_sample_feature_list,Waveform_rawdata_sample_feature_list,Spectral_rawdata_sample_feature_list,rawdata_sample_feature_list = get_feature_list(rawdata)   # 原始数据特征 12,24
#         # diffdata_sample_feature_list = get_feature_list(diffdata)  # 差分数据特征 12,24
#         # sample_feature_list = np.concatenate((rawdata_sample_feature_list, diffdata_sample_feature_list), axis=1)   # 两类特征合并 12,32
#         Time_feature_data = np.reshape(Time_rawdata_sample_feature_list, (1, Time_rawdata_sample_feature_list.shape[0], Time_rawdata_sample_feature_list.shape[1]))
#         Waveform_feature_data = np.reshape(Waveform_rawdata_sample_feature_list, (1, Waveform_rawdata_sample_feature_list.shape[0], Waveform_rawdata_sample_feature_list.shape[1]))
#         Spectral_feature_data = np.reshape(Spectral_rawdata_sample_feature_list, (1, Spectral_rawdata_sample_feature_list.shape[0], Spectral_rawdata_sample_feature_list.shape[1]))
#         feature_data = np.reshape(rawdata_sample_feature_list, (1, rawdata_sample_feature_list.shape[0], rawdata_sample_feature_list.shape[1]))
        
#         Time_temp[i, :, :] = Time_feature_data
#         Waveform_temp[i, :, :] = Waveform_feature_data
#         Spectral_temp[i, :, :] = Spectral_feature_data
#         temp[i, :, :] = feature_data
#         label = int(name_list[i].split(' ')[1])
#         label_temp[i] = label
#     # temp = temp.reshape(len(name_list), -1)  # 样本量，12*32（展平）
#     return Time_temp,Waveform_temp,Spectral_temp,temp, label_temp

# def get_stats_features(rootpath, labelpath):
    
#     datapath = rootpath
#     file = open(labelpath)
#     name_list = []
#     for f in file:
#         name_list.append(f)
#     temp = np.empty([len(name_list), 12, 16])
#     label_temp = np.empty(len(name_list))
#     for i in range(len(name_list)):
#         path = datapath + name_list[i].split(' ')[0]
#         rawdata = scio.loadmat(path)['data']              # 原始数据 10000,12
#         diffdata = get_diff_data(rawdata)               # 差分数据 9999,12
#         rawdata_sample_feature_list = get_feature_list(rawdata)   # 原始数据特征 12,16
#         diffdata_sample_feature_list = get_feature_list(diffdata)  # 差分数据特征 12,16
#         # sample_feature_list = np.concatenate((rawdata_sample_feature_list, diffdata_sample_feature_list), axis=1)   # 两类特征合并 12,32
#         feature_data = np.reshape(rawdata_sample_feature_list, (1, rawdata_sample_feature_list.shape[0], rawdata_sample_feature_list.shape[1]))
#         temp[i, :, :] = feature_data
#         label = int(name_list[i].split(' ')[1])
#         label_temp[i] = label
#     temp = temp.reshape(len(name_list), -1)  # 样本量，12*32（展平）
#     return temp, label_temp