import os, argparse, time
import numpy as np
import scipy.io as scio
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



"test synch"

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *a, **k): return it

# ---------------- affichage style "stability" ----------------
def draw_loss_acc(train_acc, train_loss, val_acc, val_loss, out_png: str) -> None:
    """Plot training/validation accuracy and loss over epochs."""
    epochs = range(len(train_acc))
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_acc,label="train")
    plt.plot(epochs, val_acc,label="val")
    plt.title('Accuracy vs. epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_loss,label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()
    plt.close()

def draw_confusion_matrix(C: np.ndarray, class_names: list, out_png: str) -> None:
    """Plot and save the confusion matrix with metrics."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    df = pd.DataFrame(C, index=class_names, columns=class_names)
    sns.heatmap(df, fmt='d', annot=True, cmap='Blues', cbar=True,
                annot_kws={'size': 10})
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()
    plt.close()
    # Résumé
    total = C.sum()
    diagonal = np.diag(C).sum()
    acc = diagonal / total if total else 0.0
    print(f"Overall accuracy: {acc:.4f}")

# -------------------- utils de structure --------------------
def _is_2d_numeric(arr: np.ndarray) -> bool:
    return isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.dtype != object

def _auto_to_LS(arr: np.ndarray) -> np.ndarray:
    # L >= S : si lignes < colonnes on transpose (S,L)->(L,S)
    return arr.T if arr.shape[0] < arr.shape[1] else arr

def _unwrap_1x1_object(x: Any) -> Any:
    obj = x
    while isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        obj = obj.item()
    return obj

def _iter_deep(obj: Any, path: str, out: List[Tuple[str, Any]], max_depth=6, depth=0):
    """Collecte (path, objet) pour tous les sous-objets, pour scan global."""
    if depth > max_depth:
        return
    out.append((path, obj))
    # Struct MATLAB
    if isinstance(obj, np.void) and hasattr(obj, "dtype") and obj.dtype.names:
        for name in obj.dtype.names:
            _iter_deep(obj[name], f"{path}.{name}", out, max_depth, depth+1)
    # Numpy array
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for i, el in enumerate(obj.ravel()):
                _iter_deep(el, f"{path}[{i}]", out, max_depth, depth+1)

def scan_gtn_struct(gtn_root: Any) -> Dict[str, Any]:
    """Scanne GTN_dataset (struct 1x1) et recense les 2D, les cells, etc."""
    acc: List[Tuple[str, Any]] = []
    _iter_deep(gtn_root, "GTN", acc)
    report = {
        "twoD": [],          # (path, shape)
        "cells": [],         # (path, shape)
        "scalars": [],       # (path, val dtype)
        "other": []          # (path, type, shape, dtype)
    }
    for p, o in acc:
        oo = _unwrap_1x1_object(o)
        if isinstance(oo, np.ndarray):
            if _is_2d_numeric(oo):
                report["twoD"].append((p, oo.shape))
            elif oo.dtype == object:
                report["cells"].append((p, oo.shape))
            elif oo.ndim == 0:
                report["scalars"].append((p, (oo.item() if oo.size else None, str(oo.dtype))))
            else:
                report["other"].append((p, type(oo).__name__, getattr(oo, "shape", None), str(getattr(oo, "dtype", None))))
        elif isinstance(oo, np.void):
            # struct : laissé à 'other'
            report["other"].append((p, "struct", None, None))
        else:
            # python scalars, lists, etc.
            shp = getattr(oo, "shape", None)
            dt = getattr(oo, "dtype", None)
            report["other"].append((p, type(oo).__name__, shp, str(dt) if dt is not None else None))
    return report

# --------------- extraction high-level ---------------
def _labels_to_vec(y) -> np.ndarray:
    y = _unwrap_1x1_object(y)
    arr = np.array(y).squeeze()
    if arr.dtype == object:
        vals = []
        for el in arr.ravel():
            el = _unwrap_1x1_object(el)
            vals.append(int(np.array(el).squeeze()))
        arr = np.array(vals, dtype=np.int64)
    else:
        arr = arr.astype(np.int64, copy=False)
    return arr

def _cell_to_2d_list(cell) -> List[np.ndarray]:
    """Convertit une cell (1,N) en liste de matrices 2D (si possible)."""
    cell = np.atleast_2d(cell)
    mats = []
    for el in cell.ravel():
        obj = _unwrap_1x1_object(el)
        if isinstance(obj, np.void) and getattr(obj, "dtype", None) is not None:  # struct
            # essaye tous les champs
            found = None
            for nm in obj.dtype.names:
                cand = _unwrap_1x1_object(obj[nm])
                cand_arr = np.array(cand)
                if cand_arr.ndim == 2 and cand_arr.dtype != object:
                    found = cand_arr
                    break
            if found is not None:
                mats.append(found)
                continue
        arr = np.array(obj)
        if arr.ndim == 2 and arr.dtype != object:
            mats.append(arr)
    return mats

def _gather_all_2d_pools(report: Dict[str, Any], gtn: Any) -> List[List[np.ndarray]]:
    """Crée une liste de 'pools' candidate (listes d'échantillons 2D) en explorant les paths des cells."""
    pools: List[List[np.ndarray]] = []
    # 1) Essayer directement des cells connues 'train', 'test'
    for name in ("train", "test", "data", "samples", "all", "X"):
        try:
            cell = gtn[name]
            cell = _unwrap_1x1_object(cell)
            if isinstance(cell, np.ndarray) and cell.dtype == object:
                mats = _cell_to_2d_list(cell)
                if len(mats) > 0:
                    pools.append(mats)
        except Exception:
            pass
    # 2) Explorer toutes les cells repérées par le scanner
    #    ATTENTION: on évalue à partir de gtn via eval path simplifié
    for p, shp in report["cells"]:
        # on ne va pas eval arbitraire; on tente de rechopper via noms simples
        # Si p = "GTN.train", on a déjà vu. On va ignorer pour sécurité.
        pass
    return pools

def load_gtn_auto(mat_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = scio.loadmat(mat_path)
    if "GTN_dataset" not in d:
        raise KeyError("GTN_dataset introuvable")
    gtn = d["GTN_dataset"]
    assert gtn.shape == (1,1), f"GTN_dataset shape={gtn.shape}"
    gtn = gtn[0,0]

    # labels (eux doivent exister)
    y_train = _labels_to_vec(gtn["trainlabels"])
    y_test  = _labels_to_vec(gtn["testlabels"])

    # Cas 1: train/test contiennent déjà des matrices 2D (cell -> 2D)
    train_cell = _unwrap_1x1_object(gtn["train"])
    test_cell  = _unwrap_1x1_object(gtn["test"])
    if isinstance(train_cell, np.ndarray) and train_cell.dtype == object:
        mats_tr = _cell_to_2d_list(train_cell)
        mats_te = _cell_to_2d_list(test_cell) if isinstance(test_cell, np.ndarray) and test_cell.dtype == object else []
        if len(mats_tr) == len(y_train) and len(mats_te) == len(y_test) and len(mats_tr) > 0:
            X_train = np.stack([_auto_to_LS(m).astype(np.float32, copy=False) for m in mats_tr], 0)
            X_test  = np.stack([_auto_to_LS(m).astype(np.float32, copy=False) for m in mats_te], 0)
            return X_train, y_train, X_test, y_test

    # Cas 2: 'train' contient des scalaires (indices) vers un pool ailleurs
    # -> on scanne le GTN pour trouver AU MOINS un pool de matrices 2D
    report = scan_gtn_struct(gtn)
    pools = _gather_all_2d_pools(report, gtn)

    # Heuristique: si un unique pool a exactement len(y_train)+len(y_test) éléments
    # on suppose que train/test sont des indices dans ce pool (0- ou 1-based).
    if pools:
        # choisir le pool le plus gros
        pool = max(pools, key=len)
        Npool = len(pool)
        # tenter de lire 'train' / 'test' comme indices
        def to_indices(obj):
            arr = _unwrap_1x1_object(obj)
            arr = np.array(arr).ravel()
            # cast en int (si float32)
            try:
                idx = arr.astype(np.int64)
            except Exception:
                # si pas castable, abort
                return None
            return idx

        idx_tr = to_indices(train_cell)
        idx_te = to_indices(test_cell)
        if idx_tr is not None and idx_te is not None:
            # tolérer 1-based
            if idx_tr.min() == 1 or idx_te.min() == 1:
                idx_tr = idx_tr - 1
                idx_te = idx_te - 1
            # borne
            if idx_tr.max() < Npool and idx_te.max() < Npool:
                mats_tr = [pool[i] for i in idx_tr]
                mats_te = [pool[i] for i in idx_te]
                X_train = np.stack([_auto_to_LS(m).astype(np.float32, copy=False) for m in mats_tr], 0)
                X_test  = np.stack([_auto_to_LS(m).astype(np.float32, copy=False) for m in mats_te], 0)
                return X_train, y_train, X_test, y_test

    # Si on arrive ici: extraction automatique impossible → rapport détaillé
    print("=== Rapport de scan GTN_dataset ===")
    print(f"twoD (exemples): {report['twoD'][:8]}")
    print(f"cells (exemples): {report['cells'][:8]}")
    print(f"scalars (exemples): {report['scalars'][:8]}")
    raise ValueError("Impossible d'extraire automatiquement des échantillons 2D depuis ce .mat.\n"
                     "Le rapport ci-dessus montre où se trouvent les matrices 2D candidates.\n"
                     "Indique-moi le champ où sont stockées les trames (liste de matrices 2D).")

# ----------------- Dataset/Training (identique) -----------------
class GTNDatasetTorch(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return {"data": torch.from_numpy(self.X[idx]).float(),
                "label": torch.tensor(self.y[idx], dtype=torch.long)}

from DASFormer import DASFormer

def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train(); run_loss=0.0; seen=0; preds=[]; labs=[]
    pbar = tqdm(loader, total=len(loader), desc="Train", unit="batch")
    for b in pbar:
        x=b["data"].to(device); y=b["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss): continue
        loss.backward()
        if max_grad_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        bs=x.size(0); run_loss+=loss.item()*bs; seen+=bs
        pred=logits.argmax(1); preds+=pred.tolist(); labs+=y.tolist()
        pbar.set_postfix({"loss":f"{loss.item():.4f}", "avg_loss":f"{run_loss/max(1,seen):.4f}",
                          "acc":f"{accuracy_score(labs,preds):.4f}"})
    return run_loss/max(1,seen), accuracy_score(labs,preds)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); run_loss=0.0; seen=0; preds=[]; labs=[]
    for b in loader:
        x=b["data"].to(device); y=b["label"].to(device)
        logits=model(x); loss=criterion(logits,y)
        if not torch.isfinite(loss): continue
        bs=x.size(0); run_loss+=loss.item()*bs; seen+=bs
        preds+=logits.argmax(1).tolist(); labs+=y.tolist()
    return run_loss/max(1,seen), accuracy_score(labs,preds), preds, labs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mat_path", type=str, default=r"D:\Michel\DAStatFormer\DAS_DataNorm_all_domains_for_GTN.mat")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=0)
    args=ap.parse_args()

    print("Loading MATLAB dataset (auto)…")
    X_train, y_train, X_test, y_test = load_gtn_auto(args.mat_path)
    print(f"Loaded: X_train={X_train.shape}, X_test={X_test.shape}")

    _, L, S = X_train.shape
    if y_train.min()==1 and y_test.min()==1:
        y_train=y_train-1; y_test=y_test-1
    num_classes = int(max(y_train.max(), y_test.max())+1)

    pin=torch.cuda.is_available()
    train_loader=DataLoader(GTNDatasetTorch(X_train,y_train), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader=DataLoader(GTNDatasetTorch(X_test,y_test), batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    t_slice_sizes=[4,4,3]; t_hidden_dims=[64,128,256]; t_num_heads=[4,4,8]; t_reductions=[2,1,1]
    s_slice_sizes=[2,2,2]; s_hidden_dims=[32,64,128]; s_num_heads=[2,4,4]; s_reductions=[2,1,1]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=DASFormer(num_classes, t_slice_sizes,t_hidden_dims,t_num_heads,t_reductions,
                    s_slice_sizes,s_hidden_dims,s_num_heads,s_reductions,
                    time_input_dim=S, space_input_dim=L).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best=-1.0; best_path="dasformer_best_gtn_auto.pth"
    hist_train_loss, hist_train_acc, hist_val_loss, hist_val_acc = [], [], [], []
    

    for epoch in range(1, args.epochs+1):
        tr_loss,tr_acc=train_one_epoch(model,train_loader,criterion,optimizer,device)
        va_loss,va_acc, preds, labs = evaluate(model,val_loader,criterion,device)
        print(f"Epoch {epoch}/{args.epochs} | Train loss {tr_loss:.4f}, acc {tr_acc:.4f} | "
              f"Val loss {va_loss:.4f}, acc {va_acc:.4f}")
        hist_train_loss.append(tr_loss); hist_train_acc.append(tr_acc)
        hist_val_loss.append(va_loss);   hist_val_acc.append(va_acc)
        
        if va_acc>best:
            best=va_acc; torch.save(model.state_dict(), best_path)
    print(f"Done. Best val acc={best:.4f}. Saved: {best_path}")
    
    # Courbes train/val
    draw_loss_acc(hist_train_acc, hist_train_loss, hist_val_acc, hist_val_loss,
                  out_png="accuracy_loss_dasformer_fixed.jpg")
    
    # =========================
    # Matrice de confusion (VAL)
    # =========================
    _, _, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    C = confusion_matrix(val_labels, val_preds)
    class_names = [str(i) for i in range(num_classes)]  # ou tes noms réels
    draw_confusion_matrix(C, class_names, out_png="DASFormer_confusion_matrix_fixed.jpg")

if __name__=="__main__":
    main()
