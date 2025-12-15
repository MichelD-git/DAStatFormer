import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# ----------------------------
# 0) Paramètres
# ----------------------------
mat_path = r"E:\Michel\GTN-master\Gated Transformer 论文IJCAI版\GTN_dataset_structured.mat"
out_dir  = r"."
os.makedirs(out_dir, exist_ok=True)

class_names = ['background', 'digging', 'knocking', 'watering', 'shaking', 'walking']
classes_order = list(range(len(class_names)))  # [0,1,2,3,4,5]

# ----------------------------
# 1) Lecture .mat
# ----------------------------
mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
gtn = mat["GTN_dataset"]

def get_field(struct, name):
    if hasattr(struct, name):
        return getattr(struct, name)
    if isinstance(struct, np.ndarray) and struct.dtype.names:
        return struct[name].squeeze()
    raise KeyError(name)

train_cells      = get_field(gtn, "train")
trainlabels_cell = get_field(gtn, "trainlabels")

def cell_to_stacked_array(cell_like, dtype=np.float32):
    if isinstance(cell_like, np.ndarray) and cell_like.dtype == np.object_:
        elems = [np.asarray(x) for x in cell_like.ravel()]
    elif isinstance(cell_like, (list, tuple)):
        elems = [np.asarray(x) for x in cell_like]
    else:
        return np.asarray(cell_like, dtype=dtype)
    return np.stack(elems).astype(dtype)

def cell_labels_to_1d(cell_like, dtype=np.int64):
    if isinstance(cell_like, np.ndarray) and cell_like.dtype == np.object_:
        vals = [np.asarray(x).squeeze() for x in cell_like.ravel()]
        arr = np.array(vals)
    else:
        arr = np.asarray(cell_like).squeeze()
    return arr.astype(dtype).ravel()

X_train = cell_to_stacked_array(train_cells, dtype=np.float32)   # (N, 24, 12)
y_train = cell_labels_to_1d(trainlabels_cell, dtype=np.int64)    # (N,)

if y_train.min() != 0:
    y_train = y_train - y_train.min()

print("X_train:", X_train.shape, "y_train:", y_train.shape, "classes:", np.unique(y_train))

# ----------------------------
# 2) Indices des attributs
# ----------------------------
idx_RMS   = 8    # RMS
idx_crest = 13   # Crest factor
idx_sent  = 23   # Spectral entropy
idx_imp   = 12   # Impulse factor
idx_senrg = 22   # Spectral energy

# ----------------------------
# 3) Agrégation (médiane par canal)
# ----------------------------
rms_per_seg   = np.median(X_train[:, idx_RMS,   :], axis=1)
imp_per_seg   = np.median(X_train[:, idx_imp,   :], axis=1)
senrg_per_seg = np.median(X_train[:, idx_senrg, :], axis=1)

df = pd.DataFrame({
    "class": y_train,
    "RMS": rms_per_seg,
    "ImpulseFactor": imp_per_seg,
    "SpectralEnergy": senrg_per_seg
})

# ----------------------------
# 4) Fonction de tracé
# ----------------------------
def plot_box(df, value_col, classes, out_pdf, y_label, title):
    data = [df.loc[df["class"] == c, value_col].dropna().values for c in classes]
    labels = [class_names[c] for c in classes]
    fig, ax = plt.subplots(figsize=(7,3.2))
    ax.boxplot(data, labels=labels, showmeans=True, whis=1.5,
               patch_artist=True,
               boxprops=dict(facecolor="lightblue", color="black"),
               medianprops=dict(color="red", linewidth=2),
               meanprops=dict(marker="D", markerfacecolor="green", markersize=5))
    ax.set_xlabel("Class"); ax.set_ylabel(y_label); ax.set_title(title)
    # ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, out_pdf), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# ----------------------------
# 5) Tracé et sauvegarde
# ----------------------------
plot_box(df, "RMS",             classes_order, "box_RMS.pdf",             "RMS",              " ")
plot_box(df, "ImpulseFactor",   classes_order, "box_impulse_factor.pdf",  "Impulse factor",    " ")
plot_box(df, "SpectralEnergy",  classes_order, "box_spectral_energy.pdf", "Spectral energy",   " ")

print("PDFs saved in", out_dir)


# Liste des noms des 24 attributs (doit suivre l’ordre de X_train[:, feat_idx, :])
feat_names = [
    "max","min","peak2peak","mean","variance","std","skew","kurtosis","RMS","energy",
    "rectified_mean","waveform_factor","impulse_factor","crest_factor","margin_factor",
    "kurtosis_factor","envelope_skewness","envelope_kurtosis",
    "mean_spectrum","var_spectrum","max_spectrum","min_spectrum","spectral_energy","spectral_entropy"
]

# Agrégation : médiane sur les 12 canaux → un scalaire par segment
agg_feats = {name: np.median(X_train[:, idx, :], axis=1) for idx, name in enumerate(feat_names)}
df_all = pd.DataFrame(agg_feats)
df_all["class"] = y_train

# Création des dossiers de sortie
out_dir = "./all_boxplots"
os.makedirs(out_dir, exist_ok=True)

def plot_box_attr(df, attr, classes, out_path):
    data = [df.loc[df["class"] == c, attr].dropna().values for c in classes]
    labels = [class_names[c] for c in classes]
    fig, ax = plt.subplots(figsize=(7,3.2))
    ax.boxplot(data, labels=labels, showmeans=True, whis=1.5,
               patch_artist=True,
               boxprops=dict(facecolor="lightblue", color="black"),
               medianprops=dict(color="red", linewidth=2),
               meanprops=dict(marker="D", markerfacecolor="green", markersize=5))
    ax.set_xlabel("Class"); ax.set_ylabel(attr); ax.set_title(attr + " by class")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# Boucle sur tous les attributs
for attr in feat_names:
    out_path = os.path.join(out_dir, f"box_{attr}.pdf")
    plot_box_attr(df_all, attr, classes_order, out_path)

print(f"Tous les boxplots enregistrés dans {out_dir}")