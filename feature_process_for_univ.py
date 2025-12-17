# -*- coding: utf-8 -*-
"""
Adaptation pour dossiers de classes:
root/
  ├─ train/
  │    ├─ car/*.mat
  │    ├─ construction/*.mat
  │    └─ ...
  └─ test/
       ├─ car/*.mat
       └─ ...

Chaque .mat doit contenir une variable 'data' 2D (T,C) ou (C,T).
Pour chaque canal: 24 features sur RAW + 24 features sur DIFF -> 48 features/canal.
On remet chaque échantillon à target_channels (par défaut 1700) le long de l'axe "canal".
"""

# Walking 
# --- change seulement les signatures + un if dans extract_48feats_from_split ---


import sys, os, datetime
from sklearn import preprocessing
import os
from typing import Dict, List, Tuple, Literal

import numpy as np
import scipy.io as scio

# ta fonction 24-features (exactement celle que tu utilises déjà)
from feature_extraction import feature_extraction,feature_extraction_selected24

from scipy.io import savemat

class Logger(object):
    def __init__(self, filename='svm_result.log', stream=sys.stdout, encoding='utf-8'):
        self.terminal = stream
        self.log = open(filename, 'w', encoding=encoding)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass

sys.stdout = Logger('svm_result.log', sys.stdout)
# -------- utils de forme / robustesse --------
def _ensure_CT(arr: np.ndarray) -> np.ndarray:
    """Force l'orientation (C, T) à partir de (T, C) ou (C, T)."""
    x = np.asarray(arr)
    if x.ndim != 2:
        raise ValueError(f"data doit être 2D, reçu {x.shape}")
    # Heuristique simple : souvent T >> C
    # On veut (C, T)
    if x.shape[0] > x.shape[1]:
        x = x.T
    return x  # (C, T)


def _diff_time(x_CT: np.ndarray) -> np.ndarray:
    """Différence 1er ordre sur l'axe temps: (C,T) -> (C,T-1)."""
    return np.diff(x_CT, axis=1)


def _resample_channels(feats_CxD: np.ndarray, target_C: int) -> np.ndarray:
    """
    Interpolation linéaire 1D sur l’axe "canal" pour obtenir (target_C, D).
    """
    C, D = feats_CxD.shape
    if C == target_C:
        return feats_CxD.astype(np.float32, copy=False)

    xs_old = np.linspace(0.0, 1.0, C, endpoint=True)
    xs_new = np.linspace(0.0, 1.0, target_C, endpoint=True)
    out = np.empty((target_C, D), dtype=np.float32)
    for j in range(D):
        out[:, j] = np.interp(xs_new, xs_old, feats_CxD[:, j].astype(np.float64))
    return out


def _features_per_channel(x_CT: np.ndarray) -> np.ndarray:
    """
    Calcule 24 features pour chaque canal d’un signal 2D (C,T).
    Retourne (C, 24).
    """
    C, _ = x_CT.shape
    feats = np.empty((C, 24), dtype=np.float32)
    for c in range(C):
        # sécurité: remplace non-finies par 0 avant feature_extraction
        row = np.asarray(x_CT[c, :], dtype=np.float64)
        row[~np.isfinite(row)] = 0.0
        feats[c, :] = feature_extraction_selected24(row).astype(np.float32)
    return feats


# -------- cœur: parcours dossier split --------
# --- change seulement les signatures + un if dans extract_48feats_from_split ---

def extract_48feats_from_split(
    split_dir: str,
    var_name: str = "x",
    target_channels: int = 1700,
    extensions=(".mat",),
    progress: bool = True,
    include_classes=None,         # <- ajout: iterable de noms de classes à inclure (ou None)
):
    import os, numpy as np, scipy.io as scio
    # liste classes présentes
    classes_all = sorted([d for d in os.listdir(split_dir)
                          if os.path.isdir(os.path.join(split_dir, d))])
    if include_classes is not None:
        include = set(include_classes)
        classes = [c for c in classes_all if c in include]
    else:
        classes = classes_all
    if not classes:
        raise FileNotFoundError(f"Aucune classe ciblée dans {split_dir}")

    label2id = {cls: i for i, cls in enumerate(classes)}
    X_list, y_list, files = [], [], []

    total_files = sum(
        len([f for f in os.listdir(os.path.join(split_dir, cls)) if f.lower().endswith(extensions)])
        for cls in classes
    )
    seen = 0

    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        mats = [f for f in os.listdir(cls_dir) if f.lower().endswith(extensions)]
        for fname in mats:
            fpath = os.path.join(cls_dir, fname)
            try:
                mdict = scio.loadmat(fpath)
                if var_name not in mdict:
                    print(f"[WARN] '{var_name}' absent -> skip: {fpath}")
                    continue

                x = _ensure_CT(mdict[var_name]).astype(np.float64, copy=False)
                x[~np.isfinite(x)] = 0.0

                feats_raw  = _features_per_channel(x)                             # (C,24)
                # x_diff     = np.diff(x, axis=1) if x.shape[1] > 1 else x
                # feats_diff = _features_per_channel(x_diff)                        # (C,24)
                feats      = feats_raw #np.concatenate([feats_raw, feats_diff], axis=1)      # (C,48)
                feats_tgt  = _resample_channels(feats, target_channels)           # (target_C,48)

                X_list.append(feats_tgt.astype(np.float32, copy=False))
                y_list.append(label2id[cls])
                files.append(fpath)
            except Exception as e:
                print(f"[ERROR] {fpath}: {e}")

            seen += 1
            if progress and (seen % 50 == 0 or seen == total_files):
                print(f"  progress: {seen}/{total_files}")

    if not X_list:
        raise RuntimeError(f"Aucun échantillon valide trouvé dans {split_dir} pour {include_classes}")

    X = np.stack(X_list, axis=0)         # (N, target_C, 48)
    y = np.array(y_list, dtype=np.int64) # (N,)
    return X, y, label2id, files


train_dir = r"D:\Michel\DAS-dataset\New_data_splitted\train"
test_dir  = r"D:\Michel\DAS-dataset\New_data_splitted\test"

Xtr_w, ytr_w, map_tr, files_tr = extract_48feats_from_split(
    split_dir=train_dir,
    var_name="x",
    target_channels=1700,
    include_classes=None,   # <- filtre
)
Xte_w, yte_w, map_te, files_te = extract_48feats_from_split(
    split_dir=test_dir,
    var_name="x",
    target_channels=1700,
    include_classes=None   # <- filtre
)

# print("train walking:", Xtr_w.shape, ytr_w.shape)
# print("test  walking:", Xte_w.shape, yte_w.shape)

# yte_8 = yte_w+8
# ytr_8 = ytr_w+8

# -------- façade style “get_das_data” --------
def get_das_data(
    root_split_dir: str,
    var_name: str = "x",
    target_channels: int = 1700,
    flatten: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Version API compatible avec ton ancien get_das_data(...) :
      - root_split_dir: dossier 'train' OU 'test' (pas le parent).
      - retourne (temp, label_temp)
        * temp : (N, target_channels*48) si flatten=True, sinon (N, target_channels, 48)
        * label_temp : (N,) entiers (ordre alphabétique des dossiers de classes)
    """
    X, y, _, _ = extract_48feats_from_split(
        split_dir=root_split_dir,
        var_name=var_name,
        target_channels=target_channels,
    )
    if flatten:
        N, C, D = X.shape
        X = X.reshape(N, C * D)
    return X, y


# Exemple
ROOT = r"D:\Michel\DAS-dataset\New_data_splitted" # r"D:\Michel\DAS-dataset\data_test" #

# train
X_train, y_train = get_das_data(os.path.join(ROOT, "train"),
                                var_name="x",
                                target_channels=1700,
                                flatten=True)   # (N, 1700, 48)

# test
X_test, y_test = get_das_data(os.path.join(ROOT, "test"),
                              var_name="x",
                              target_channels=1700,
                              flatten=True)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# ---------- Chargement ----------
# start_train = datetime.datetime.now()
# X_train, y_train = get_das_data(train_rootpath, train_labelpath)
# X_test, y_test   = get_das_data(test_rootpath,  test_labelpath)

# # ---------- Scaling sans fuite ----------
scaler = preprocessing.MinMaxScaler()
trainingData = scaler.fit_transform(X_train)
testData     = scaler.transform(X_test)

# ---------- Sauvegardes CSV (avec en-têtes) ----------
# def save_with_labels_csv(X_scaled, y, out_csv):
#     df = pd.DataFrame(X_scaled)
#     df['label'] = y
#     df.to_csv(out_csv, index=False)

# save_with_labels_csv(trainingData, y_train, 'GTN_trainingData_Time_1D.csv')
# save_with_labels_csv(testData,     y_test,   'GTN_TestingData_Time_1D.csv')

# ---------- Reshape cohérent (C, T) ----------
def choose_channels(n_features, candidates=(1700, 24)):
    """
    Choisit C parmi 'candidates' tel que n_features % C == 0 ; sinon lève une erreur explicite.
    """
    for C in candidates:
        if n_features % C == 0:
            return C
    raise ValueError(f"Impossible de factoriser {n_features} en (C,T) avec C dans {candidates}.")

def reshape_CT(X, C):
    T = X.shape[1] // C
    if C * T != X.shape[1]:
        raise ValueError(f"Reshape impossible: {X.shape[1]} features != {C}*T.")
    return X.reshape(X.shape[0], C, T)

# n_features_train = trainingData.shape[1]
# n_features_test  = testData.shape[1]

# On impose **le même C** pour train et test



# train_dataset = pd.read_csv(r'C:\Users\michel\Downloads\Phi-OTDR_dataset_and_codes-main\GTN_trainingData_all_domains_1D.csv')

# Xtrain = np.array(train_dataset)[:,0:288]
# ytrain = np.array(train_dataset)[:,-1]



# test_dataset = pd.read_csv(r'C:\Users\michel\Downloads\Phi-OTDR_dataset_and_codes-main\GTN_TestingData_all_domains_1D.csv')
# Xtest = np.array(test_dataset)[:,0:288]
# ytest = np.array(test_dataset)[:,-1]

# trainingData = X_train
# y_train  = y_train
# testData = X_test
# y_test = y_test

n_features_train = trainingData.shape[1]
n_features_test  = testData.shape[1]

# On impose **le même C** pour train et test
C = choose_channels(n_features_train, candidates=(1700, 24))
if n_features_test % C != 0:
    # Dernière chance: si train et test n'ont pas le même nb de features, on échoue explicitement
    raise ValueError(f"Incohérence features: train={n_features_train}, test={n_features_test} non divisible par C={C}.")

X_train_3D = reshape_CT(trainingData, C)   # (N, C, T)
X_test_3D  = reshape_CT(testData, C)       # (N, C, T)
# ---------- Conversions pour .mat ----------
def _to_cell_samples(X, ensure_c_first=None):
    """
    X: array (N, C, T) ou liste de (C,T).
    Retour: cell array MATLAB shape (1, N), chaque cellule (C, T) float32.
    """
    X = np.asarray(X)
    assert X.ndim == 3, "Attendu X de rang 3 (N, C, T)."
    N, Cc, Tt = X.shape
    cells = np.empty((1, N), dtype=object)
    for i in range(N):
        xi = X[i]
        if ensure_c_first is not None and xi.shape[0] != ensure_c_first and xi.shape[1] == ensure_c_first:
            xi = xi.T
        cells[0, i] = np.array(xi, dtype=np.float32)
    return cells

def _to_numeric_row(y):
    y = np.asarray(y).reshape(-1).astype(np.int64, copy=False)
    return y.reshape(1, -1)

def save_for_MyDataset3d(
    X_train3d, y_train, X_test3d, y_test,
    out_path="GTN_dataset.mat",
    channels_first=None,
    key_name="GTN_dataset"
):
    train_cells = _to_cell_samples(X_train3d, ensure_c_first=channels_first)
    test_cells  = _to_cell_samples(X_test3d,  ensure_c_first=channels_first)
    trainlabels = _to_numeric_row(y_train)   # (1, Ntrain)
    testlabels  = _to_numeric_row(y_test)    # (1, Ntest)

    dtype = [('trainlabels','O'), ('train','O'), ('testlabels','O'), ('test','O')]
    gtn = np.empty((1, 1), dtype=dtype)
    gtn['trainlabels'][0,0] = trainlabels
    gtn['train'][0,0]       = train_cells
    gtn['testlabels'][0,0]  = testlabels
    gtn['test'][0,0]        = test_cells

    savemat(out_path, {key_name: gtn}, do_compression=True)

# Sauvegarde .mat avec le **C** détecté
save_for_MyDataset3d(
    X_train_3D, y_train,
    X_test_3D,  y_test,
    out_path=r"D:\Michel\DAS-dataset\DAS_Data_univ_for_DAStatFormer_24.mat",
    channels_first=C,           # aligne la convention (C, T)
    key_name="GTN_dataset"
)

# ---------- Fin propre du logger ----------
if isinstance(sys.stdout, Logger):
    sys.stdout.close()
    sys.stdout = sys.__stdout__

print(f"Terminé. C={C}, train shape={X_train_3D.shape}, test shape={X_test_3D.shape}")
