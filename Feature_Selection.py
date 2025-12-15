import numpy as np
import pandas as pd
from scipy.io import loadmat
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from pathlib import Path

# === 0) Utilitaire pour dépaqueter objets MATLAB ===
def _to_numpy_array(obj, expect_feature_shape=None, name="array"):
    """
    Convertit un array dtype=object (cell array MATLAB) en ndarray numérique.
    - Si obj est déjà un ndarray float, le retourne.
    - Si obj est (1,1) struct-wrapper, essaye d'accéder à [0,0].
    - Si obj est un array d'objets (n,) où chaque élément est une matrice 12x24,
      on empile le tout en (n, 12, 24).
    """
    arr = obj
    # Cas struct MATLAB encapsulé: shape (1,1) avec dtype=object contenant un ndarray
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
        arr = arr.item()  # déballe le seul élément

    # Si déjà numérique
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        if expect_feature_shape and arr.ndim == 2 and arr.shape == expect_feature_shape:
            # cas rare: un seul échantillon; on ajoute l'axe batch
            arr = arr[None, ...]
        return arr

    # Sinon, on est (très probablement) dans un cell array: array object 1D/2D
    # On le "squeeze", puis on empile chaque cellule convertie en float32
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    # Si c'est un unique élément objet (ex: (12,24)), déballe
    if arr.dtype == object and arr.ndim == 0:
        arr = np.array(arr.item(), dtype=np.float32)
        if expect_feature_shape and arr.shape == expect_feature_shape:
            arr = arr[None, ...]
        return arr

    # Sinon, itère et empile
    elems = []
    for i in range(arr.size):
        x = arr.flat[i]
        x = np.array(x, dtype=np.float32)  # chaque cellule -> ndarray float32
        elems.append(x)
    out = np.stack(elems, axis=0)

    if expect_feature_shape and out.shape[1:] != expect_feature_shape:
        raise ValueError(f"{name}: forme attendue {expect_feature_shape}, obtenu {out.shape[1:]}")
    return out

def _to_1d_labels(obj, name="labels"):
    """Convertit labels MATLAB (souvent objet/cell) en vecteur 1D np.int64."""
    lab = obj
    if isinstance(lab, np.ndarray) and lab.dtype == object and lab.size == 1:
        lab = lab.item()

    lab = np.asarray(lab).squeeze()
    # Si encore objet/cell -> convertir élément par élément
    if lab.dtype == object:
        vals = []
        for i in range(lab.size):
            v = lab.flat[i]
            if isinstance(v, (list, tuple, np.ndarray)):
                v = np.asarray(v).squeeze()
                # s'il reste >1 élément, on prend le premier (à ajuster selon ton format)
                v = v.item() if np.size(v) == 1 else v.flat[0]
            vals.append(np.int64(v))
        lab = np.asarray(vals, dtype=np.int64)
    else:
        lab = lab.astype(np.int64, copy=False)
    return lab

# === 1) Charger le .mat et repérer les champs ===
path = r'D:\Michel\Gated Transformer 论文IJCAI版\DAS_DataNorm_all_domains_for_Feature_selection.mat'
mat = loadmat(path, squeeze_me=False, struct_as_record=False)

# Repérer le struct racine (ignore les clés __header__/__version__/__globals__)
top_keys = [k for k in mat.keys() if not k.startswith('__')]
# Plusieurs cas:
# a) les champs sont au top-level: 'train', 'trainlabels', ...
# b) un struct unique au top-level qui contient ces champs.
train = trainlabels = test = testlabels = None

if {'train','trainlabels','test','testlabels'}.issubset(top_keys):
    train = mat['train']
    trainlabels = mat['trainlabels']
    test = mat['test']
    testlabels = mat['testlabels']
else:
    # Tenter: un unique struct
    if len(top_keys) == 1:
        S = mat[top_keys[0]]  # struct MATLAB -> ndarray dtype=object shape (1,1)
        # Accès de type S['train'][0,0] si c'est un structured np.array
        try:
            train = S['train'][0,0]
            trainlabels = S['trainlabels'][0,0]
            test = S['test'][0,0]
            testlabels = S['testlabels'][0,0]
        except Exception:
            # autre style: objet avec attributs (scipy old-style)
            S = S.item() if isinstance(S, np.ndarray) and S.size==1 else S
            train = getattr(S, 'train', None)
            trainlabels = getattr(S, 'trainlabels', None)
            test = getattr(S, 'test', None)
            testlabels = getattr(S, 'testlabels', None)
    else:
        raise KeyError(f"Je ne trouve pas les champs train/trainlabels au top-level: {top_keys}")

# === 2) Convertir en ndarrays Python propres ===
X_train = _to_numpy_array(train, expect_feature_shape=(12, 24), name="train")
y_train = _to_1d_labels(trainlabels, name="trainlabels")

# (Optionnel) test
if test is not None and testlabels is not None:
    X_test  = _to_numpy_array(test,  expect_feature_shape=(12, 24), name="test")
    y_test  = _to_1d_labels(testlabels, name="testlabels")
else:
    X_test = y_test = None

# Sanity checks
assert X_train.ndim == 3 and X_train.shape[1:] == (12,24), f"X_train shape invalide: {X_train.shape}"
assert y_train.ndim == 1 and y_train.shape[0] == X_train.shape[0], "y_train ne correspond pas à X_train"

print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)

# === 3) ANOVA 2-way par feature (label + channel + interaction) ===
n_samples, n_channels, n_features = X_train.shape
pvals_label, pvals_channel, pvals_inter = [], [], []
F_label, F_channel, F_inter = [], [], []

for f_idx in range(n_features):
    values = X_train[:, :, f_idx].reshape(-1)                 # (n_samples*12,)
    channels = np.repeat(np.arange(n_channels), n_samples)    # 0..11
    labs = np.tile(y_train, n_channels)

    df = pd.DataFrame({
        'value': values.astype(float),
        'channel': pd.Categorical(channels),
        'label': pd.Categorical(labs)
    })

    model = ols('value ~ C(label) + C(channel) + C(label):C(channel)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    F_label.append(anova_table.loc['C(label)', 'F'])
    F_channel.append(anova_table.loc['C(channel)', 'F'])
    F_inter.append(anova_table.loc['C(label):C(channel)', 'F'])

    pvals_label.append(anova_table.loc['C(label)', 'PR(>F)'])
    pvals_channel.append(anova_table.loc['C(channel)', 'PR(>F)'])
    pvals_inter.append(anova_table.loc['C(label):C(channel)', 'PR(>F)'])

# Correction FDR sur l’effet "label"
reject, pvals_label_fdr, _, _ = multipletests(pvals_label, alpha=0.05, method='fdr_bh')

results_df = pd.DataFrame({
    'feature_idx': np.arange(n_features),
    'F_label': F_label,
    'p_label': pvals_label,
    'p_label_fdr': pvals_label_fdr,
    'sig_label_fdr@0.05': reject,
    'F_channel': F_channel,
    'p_channel': pvals_channel,
    'F_interaction': F_inter,
    'p_interaction': pvals_inter,
}).sort_values(['sig_label_fdr@0.05','p_label_fdr','F_label'], ascending=[False,True,False]).reset_index(drop=True)

print(results_df.head(10))

# Sauvegarde
out_csv = Path(path).with_suffix('').as_posix() + '_anova2way_results.csv'
results_df.to_csv(out_csv, index=False)
print(f"Résultats ANOVA sauvegardés -> {out_csv}")
