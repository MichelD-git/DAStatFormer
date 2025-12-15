# # -*- coding: utf-8 -*-
# """
# Created on Wed Oct 15 09:53:50 2025

# @author: michel
# """


import os
from pathlib import Path
import numpy as np
import h5py
from scipy.io import savemat
from datetime import datetime
from typing import Optional, Tuple

# ---------- Config ----------
raw_dir     = r"D:\Michel\DAS-dataset\data\walking"       # dossier des .h5
bitmap_dir  = r"D:\Michel\DAS-dataset\data\walking"       # dossier des .npy
out_root    = r"D:\Michel\DAS-dataset\New_data"        # où écrire /running et /background
label_name  = "walking"                                   # étiquette
save_negatives = False                                # True -> sauver aussi background
prefix      = "walking"                                   # préfixe des .mat

# ---------- Utilitaires ----------

def find_first_dataset_2d(f: h5py.File) -> Tuple[str, Tuple[int,int]]:
    """
    Retourne (key, shape) du premier dataset 2D trouvé.
    Lève ValueError si rien de 2D.
    """
    found = []
    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
            found.append((name, obj.shape))
    f.visititems(_visit)
    if not found:
        raise ValueError("Aucun dataset 2D trouvé dans le .h5")
    # Heuristique : prendre celui avec le plus grand nombre d'éléments
    found.sort(key=lambda x: (x[1][0]*x[1][1]), reverse=True)
    return found[0]

def resize_bitmap_cols_nearest(bitmap: np.ndarray, target_C: int) -> np.ndarray:
    """
    Redimensionne les colonnes de bitmap à target_C par nearest-neighbor,
    sans dépendances externes.
    bitmap: (Nw, Cb) -> (Nw, target_C)
    """
    Nw, Cb = bitmap.shape
    if Cb == target_C:
        return bitmap
    # map indices de [0..target_C-1] -> indices origine [0..Cb-1]
    src_idx = np.round(np.linspace(0, Cb - 1, target_C)).astype(int)
    return bitmap[:, src_idx]

def edges_by_windows(T: int, Nw: int) -> np.ndarray:
    """Bornes [0..T] en Nw segments quasi égaux (len = Nw+1)."""
    if Nw <= 0 or T <= 0:
        raise ValueError(f"T={T}, Nw={Nw} invalides")
    return np.linspace(0, T, num=Nw + 1, dtype=int)

def has_event(line) -> bool:
    """True/1/float>0 -> événement présent."""
    # Supporte bool, int, float
    return bool(np.any(line > 0))

def save_window_mat(path: Path, win: np.ndarray, label: str, meta: dict):
    """Sauve une fenêtre en .mat avec métadonnées utiles."""
    savemat(path.as_posix(), {
        "x": win,                 # (twin, C)
        "label": label,
        **meta
    })

def process_pair(raw_path: Path, bitmap_path: Path, out_root: Path,
                  label_name: str, save_negatives: bool, prefix: str) -> Tuple[int,int]:
    """
    Traite un couple (raw .h5, bitmap .npy). Retourne (n_pos, n_neg_sauvegardées).
    """
    n_pos = 0
    n_neg = 0
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    out_running = out_root / label_name
    out_bg  = out_root / "background"
    out_running.mkdir(parents=True, exist_ok=True)
    if save_negatives:
        out_bg.mkdir(parents=True, exist_ok=True)

    # 1) Ouvrir le .h5 et trouver le dataset 2D principal
    with h5py.File(raw_path, "r") as hf:
        ds_key, (T, C) = find_first_dataset_2d(hf)
        dset = hf[ds_key]

        # 2) Charger la bitmap (booléen ou 0/1)
        bitmap = np.load(bitmap_path)
        if bitmap.ndim != 2:
            raise ValueError(f"Bitmap doit être 2D, reçu {bitmap.shape}")
        Nw, Cb = bitmap.shape

        # 3) Aligner colonnes si besoin
        if Cb != C:
            print(f"[WARN] Colonnes diffèrent pour {raw_path.name}: raw C={C}, bitmap Cb={Cb}. Resize bitmap -> {C}")
            bitmap = resize_bitmap_cols_nearest(bitmap, C)

        # 4) Préparer découpage
        edges = edges_by_windows(T, Nw)

        # 5) Boucle fenêtres (lecture par tranches, sans charger tout le raw)
        for i in range(Nw):
            a, b = int(edges[i]), int(edges[i+1])
            if b <= a:
                continue  # sécurité
            # lecture slice [a:b, :]
            win = dset[a:b, :]  # NumPy array (h5py lit en mémoire ce slice)

            positive = has_event(bitmap[i])
            label = label_name if positive else "background"
            base = f"{prefix}_{ts}_{raw_path.stem}_win{i:04d}_t{a}-{b}_C{C}"
            out_path = (out_running if positive else out_bg) / f"{base}.mat"

            if positive or save_negatives:
                meta = dict(
                    file_raw=raw_path.name,
                    dataset_key=ds_key,
                    window_index=int(i),
                    time_start=int(a),
                    time_end=int(b),
                    channels=int(C),
                    bitmap_cols=int(bitmap.shape[1]),
                )
                save_window_mat(out_path, win, label, meta)
                n_pos += int(positive)
                n_neg += int((not positive) and save_negatives)

    return n_pos, n_neg

def match_bitmap_for_raw(raw_path: Path, bitmap_dir: Path) -> Optional[Path]:
    """
    Trouve la bitmap .npy correspondant au .h5 par son stem.
    Ex: 'auto_2023-...+0100.h5' -> 'auto_2023-...+0100.npy'
    """
    cand = bitmap_dir / (raw_path.stem + ".npy")
    return cand if cand.exists() else None

# ---------- Lancement batch ----------
def run_batch(raw_dir: str, bitmap_dir: str, out_root: str,
              label_name="running", save_negatives=False, prefix="running"):
    raw_dir = Path(raw_dir)
    bitmap_dir = Path(bitmap_dir)
    out_root = Path(out_root)

    all_h5 = sorted(raw_dir.glob("*.h5"))
    if not all_h5:
        print(f"Aucun .h5 trouvé dans {raw_dir}")
        return

    total_pos = total_neg = 0
    for raw_path in all_h5:
        bmp_path = match_bitmap_for_raw(raw_path, bitmap_dir)
        if bmp_path is None:
            print(f"[SKIP] Pas de bitmap pour {raw_path.name}")
            continue
        try:
            n_pos, n_neg = process_pair(raw_path, bmp_path, out_root,
                                        label_name, save_negatives, prefix)
            print(f"[OK] {raw_path.name} -> pos={n_pos}, neg={n_neg}")
            total_pos += n_pos
            total_neg += n_neg
        except Exception as e:
            print(f"[ERROR] {raw_path.name}: {e}")

    print(f"\n=== RÉSUMÉ ===\nFenêtres positives sauvées: {total_pos}")
    if save_negatives:
        print(f"Fenêtres négatives sauvées: {total_neg}")

# ---- Exécution ----
run_batch(r"D:\Michel\DAS-dataset\data\walking", r"D:\Michel\DAS-dataset\data\walking" , out_root, label_name=label_name,
          save_negatives=save_negatives, prefix=prefix)








import os, random, math, shutil, csv
from pathlib import Path

# ========= CONFIG À ADAPTER =========
SRC_ROOT = Path(r"D:\Michel\DAS-dataset\New_data")        # dossiers par classe
DST_ROOT = Path(r"D:\Michel\DASFormer\New_data_splitted")          # dossier cible (structure voulue)
TRAIN_PCT = 0.80                                          # 80/20
EXTS = {".mat"}                                           # fichiers pris en compte
MOVE_FILES = False                                        # False = copie, True = déplacement
SEED = 42
# ====================================

random.seed(SEED)
(DST_ROOT / "train").mkdir(parents=True, exist_ok=True)
(DST_ROOT / "test").mkdir(parents=True, exist_ok=True)

def files_in_class(cdir: Path):
    fs = [p for p in cdir.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    fs.sort()
    random.shuffle(fs)
    return fs

def split_index(n, ratio):
    k = math.floor(n * ratio)
    if n >= 2 and k == n:  # garantit au moins 1 en test s'il y a >=2 fichiers
        k = n - 1
    return k

op = shutil.move if MOVE_FILES else shutil.copy2
summary = []
manifest = DST_ROOT / "manifest_80_20.csv"
with open(manifest, "w", newline="", encoding="utf-8") as mf:
    w = csv.writer(mf)
    w.writerow(["split","class","src","dst"])

    classes = [d for d in SRC_ROOT.iterdir() if d.is_dir()]
    total_train = total_test = 0

    for cdir in classes:
        cls = cdir.name
        files = files_in_class(cdir)
        n = len(files)
        (DST_ROOT/"train"/cls).mkdir(parents=True, exist_ok=True)
        (DST_ROOT/"test"/cls).mkdir(parents=True, exist_ok=True)

        if n == 0:
            print(f"[WARN] Classe vide: {cls}")
            summary.append((cls,0,0,0))
            continue

        k = split_index(n, TRAIN_PCT)
        train_files, test_files = files[:k], files[k:]

        for src in train_files:
            dst = (DST_ROOT/"train"/cls/src.name)
            op(src.as_posix(), dst.as_posix())
            w.writerow(["train", cls, src.as_posix(), dst.as_posix()])

        for src in test_files:
            dst = (DST_ROOT/"test"/cls/src.name)
            op(src.as_posix(), dst.as_posix())
            w.writerow(["test", cls, src.as_posix(), dst.as_posix()])

        summary.append((cls, n, len(train_files), len(test_files)))
        total_train += len(train_files); total_test += len(test_files)
        print(f"[OK] {cls:15s}  total={n:5d}  train={len(train_files):5d}  test={len(test_files):5d}")

print("\n=== RÉSUMÉ ===")
for cls, n, tr, te in summary:
    print(f"{cls:15s}  total={n:5d}  train={tr:5d}  test={te:5d}")
print(f"\nGlobal: train={total_train}  test={total_test}")
print(f"Manifeste -> {manifest}")


# from scipy.io import loadmat

# # Chemin vers ton fichier .mat
# data = loadmat("D:\Michel\DAS-dataset\New_data\car\car_20251015T150240_auto2_2023-04-17T124510+0100_win0067_t137516-139569_C1700")

# # Afficher les clés disponibles
# print(data.keys())

# # Exemple : accéder à une variable
# X = data["nom_de_variable"]
# print(X.shape)
