# -*- coding: utf-8 -*-
"""
Training with domain wrapper for DAS x=(B,T,F):
 - Normalisation automatique des axes pour garantir (B, 12, 48) ou (B, 12, 24)
 - RAW+DIFF (F>=48): time = 0:11 + 24:35 -> 22 | wave = 11:19 + 35:43 -> 16 | spec = 19:24 + 43:48 -> 10
 - RAW seul (F==24): time = 0:11 -> 11 | wave = 11:19 -> 8 | spec = 19:24 -> 5
 - Core: Transformer_4d(d_temp, d_wave, d_spec, ...)
 - Logs: gates, F1/Precision/Recall par classe
 - Robustesse: accuracy vs bruit
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset_process.dataset_process import MyDataset_3d
from module.loss import Myloss
from transformer import Transformer_4d
from utils.random_seed import setup_seed
from utils.visualization import result_visualization

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from time import time
from collections import defaultdict

# ==== Hyperparamètres & seed ====
setup_seed(30)
EPOCH = 100
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {DEVICE}')

# === Dimensions & flags (transformer) ===
d_model = 256
d_hidden = 128
q = 8
v = 8
h = 8
N = 8
dropout = 0.4
pe = True
mask = True

# === Chemins ===
reslut_figure_path = r'D:\Michel\Gated Transformer 论文IJCAI版\results_figure'
path = r'D:\Michel\Gated Transformer 论文IJCAI版\DAS_DataNorm_all_domains_for_GTN.mat'
test_interval = 5
draw_key = 1
file_name = path.split('\\')[-1][0:path.split('\\')[-1].index('.')]
optimizer_name = 'Adam'
save_dir = r'D:\Michel\Gated Transformer 论文IJCAI版\saved_model'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(reslut_figure_path, exist_ok=True)

# ==== Utils ====
def unpack_logits(output):
    """Accepte Tensor, tuple ou dict et retourne toujours les logits (B, num_classes)."""
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, dict):
        return output.get("logits", output)
    return output

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔍 Total parameters       : {total_params:,}")
    print(f"🧠 Trainable parameters   : {trainable_params:,}")
    print(f"💾 Estimated model size   : {trainable_params * 4 / 1024 ** 2:.2f} MB (float32)\n")

def normalize_TF(x: torch.Tensor) -> torch.Tensor:
    """
    Garanti x -> (B, T, F) avec F en {24, 48}.
    Si on reçoit (B, 48, 12) (axes inversés), transpose -> (B, 12, 48).
    """
    assert x.dim() == 3
    B, T, F = x.shape
    if F in (24, 48):
        return x
    if T in (24, 48) and F in (11, 12):
        return x.transpose(1, 2).contiguous()
    if F < T:
        return x.transpose(1, 2).contiguous()
    return x

# ---------- Gate logger ----------
class GateLogger:
    """
    Supporte deux formats de sortie du core:
    - Tensor (B,4) -> temp,wave,spec,chan
    - Liste de 3 Tensors (B,2) -> gates par branche (step/chan)
    """
    def __init__(self):
        self.per_epoch = []  # list of dicts {label: mean_gate}

    def log_from_output_tuple(self, out):
        # Tentative format (B,4)
        gate = out[-1]
        if isinstance(gate, torch.Tensor) and gate.dim() == 2 and gate.size(1) == 4:
            g = gate.detach().softmax(dim=-1).mean(0).cpu().numpy()
            self.per_epoch.append({"temp":float(g[0]),"wave":float(g[1]),"spec":float(g[2]),"chan":float(g[3])})
            return
        # Tentative liste de 3 gates (B,2)
        if isinstance(gate, (list, tuple)) and len(gate) == 3:
            vals = {}
            names = ["temp", "wave", "spec"]
            for name, g in zip(names, gate):
                g2 = g.detach().softmax(dim=-1).mean(0).cpu().numpy()  # (2,)
                vals[f"{name}_step"] = float(g2[0])
                vals[f"{name}_chan"] = float(g2[1])
                vals[f"{name}_branch"] = float(g2.mean())
            self.per_epoch.append(vals)

    def plot(self, save_path):
        if not self.per_epoch:
            print("[GateLogger] nothing to plot.")
            return
        keys = sorted(self.per_epoch[0].keys())
        X = np.arange(1, len(self.per_epoch)+1)
        plt.figure(figsize=(10,5))
        for k in keys:
            plt.plot(X, [d[k] for d in self.per_epoch], label=k)
        plt.xlabel("epoch"); plt.ylabel("gate weight (mean)")
        plt.title("Gate weights over epochs")
        plt.legend(loc="best", ncol=2)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.show(); plt.close()

# ---------- Per-class metrics tracker ----------
class ClassMetrics:
    """Stocke précision / rappel / F1 par classe à chaque epoch (sur validation)."""
    def __init__(self, class_labels):
        self.class_labels = class_labels
        self.history = defaultdict(list)

    def update_from_preds(self, y_true, y_pred):
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        for i in range(len(self.class_labels)):
            self.history[f"precision_{i}"].append(float(p[i]))
            self.history[f"recall_{i}"].append(float(r[i]))
            self.history[f"f1_{i}"].append(float(f1[i]))

    def plot(self, out_dir):
        if not self.history:
            print("[ClassMetrics] nothing to plot.")
            return
        os.makedirs(out_dir, exist_ok=True)
        X = np.arange(1, len(next(iter(self.history.values())))+1)

        # F1
        plt.figure(figsize=(10,5))
        for i, name in enumerate(self.class_labels):
            plt.plot(X, self.history[f"f1_{i}"], label=f"F1:{name}")
        plt.xlabel("epoch"); plt.ylabel("F1 (validation)")
        plt.title("Per-class F1 over epochs")
        plt.legend(loc="best", ncol=2); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_class_F1.png"), dpi=160)
        plt.show(); plt.close()

        # Precision
        plt.figure(figsize=(10,5))
        for i, name in enumerate(self.class_labels):
            plt.plot(X, self.history[f"precision_{i}"], label=f"Prec:{name}")
        plt.xlabel("epoch"); plt.ylabel("Precision (validation)")
        plt.title("Per-class Precision over epochs")
        plt.legend(loc="best", ncol=2); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_class_Precision.png"), dpi=160)
        plt.show(); plt.close()

        # Recall
        plt.figure(figsize=(10,5))
        for i, name in enumerate(self.class_labels):
            plt.plot(X, self.history[f"recall_{i}"], label=f"Recall:{name}")
        plt.xlabel("epoch"); plt.ylabel("Recall (validation)")
        plt.title("Per-class Recall over epochs")
        plt.legend(loc="best", ncol=2); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_class_Recall.png"), dpi=160)
        plt.show(); plt.close()

# ---------- Robustesse : bruit additif ----------
def evaluate_with_noise(model, dataloader, device, std_list=(0.0, 0.01, 0.02, 0.05, 0.1)):
    """Ajoute un bruit gaussien N(0, std * data_std) sur x complet; retourne dict {std: acc%} et trace la courbe."""
    results = {}
    model.eval()
    with torch.no_grad():
        # Estime l'écart-type global des features sur 1 batch
        data_std = None
        for xb, _ in dataloader:
            xb = normalize_TF(xb)
            data_std = xb.float().std().item()
            break
        if data_std is None: data_std = 1.0

        for s in std_list:
            correct = total = 0
            for xb, yb in dataloader:
                xb = normalize_TF(xb).to(device)
                yb = yb.to(device)
                x_noisy = xb.float() + torch.randn_like(xb.float()) * (s * data_std)
                out = model(x_noisy, 'test')
                logits = unpack_logits(out)
                pred = logits.argmax(dim=1)
                total += yb.size(0)
                correct += (pred == yb.long()).sum().item()
            results[s] = round(100.0 * correct / max(total, 1), 2)

    # plot
    xs = sorted(results.keys()); ys = [results[x] for x in xs]
    plt.figure(figsize=(7,4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel("noise std (× data_std)")
    plt.ylabel("Accuracy (%)")
    plt.title("Noise robustness")
    plt.grid(True, ls='--', alpha=.4)
    plt.tight_layout()
    plt.savefig(os.path.join(reslut_figure_path, "robustness_noise_curve.png"), dpi=160)
    plt.show(); plt.close()
    print("Robustness (noise std -> acc%):", results)
    return results

# ==== Datasets & Dataloaders (split 90/10) ====
trainval_dataset = MyDataset_3d(path, 'train')  # doit produire x: (T, F) mais on normalise au besoin
test_dataset     = MyDataset_3d(path, 'test')

val_ratio  = 0.10
train_size = int((1 - val_ratio) * len(trainval_dataset))
val_size   = len(trainval_dataset) - train_size
g = torch.Generator().manual_seed(42)
train_subset, val_subset = random_split(trainval_dataset, [train_size, val_size], generator=g)

train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_dataloader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# Dimensions globales (issues du dataset) — informatif
DATA_LEN  = trainval_dataset.train_len
d_input   = trainval_dataset.input_len
d_channel = trainval_dataset.channel_len
d_output  = trainval_dataset.output_len

# =========================
#       WRAPPER
# =========================
class DASDomainWrapper(nn.Module):
    """
    Découpe x=(B,T,F) en 3 domaines et appelle un core 3-branches.
    Si use_diff=True et F>=offset+24, concat RAW + DIFF pour chaque domaine.
    """
    def __init__(self, core_3branch: nn.Module,
                 time_idx=(0,11), wave_idx=(11,19), spec_idx=(19,24),
                 diff_offset=24, expect_T=None, expect_F=None, use_diff=True):
        super().__init__()
        self.core = core_3branch
        self.time_idx = time_idx
        self.wave_idx = wave_idx
        self.spec_idx = spec_idx
        self.diff_offset = diff_offset
        self.expect_T = expect_T
        self.expect_F = expect_F
        self.use_diff = use_diff

    def _slice_domain(self, x, start, end):
        raw = x[:, :, start:end]  # (B,T,f_raw)
        if self.use_diff and x.size(-1) >= self.diff_offset + 24:
            diff = x[:, :, self.diff_offset + start : self.diff_offset + end]
            return torch.cat([raw, diff], dim=-1)  # (B,T,f_raw+f_diff)
        return raw

    def forward(self, x, stage: str = 'train'):
        # Normalise orientation -> (B,T,F) avec F en {24,48}
        x = normalize_TF(x)
        assert x.dim() == 3, f"Attendu x=(B,T,F), reçu {tuple(x.shape)}"
        B, T, F = x.shape
        if self.expect_T is not None:
            assert T == self.expect_T, f"T attendu={self.expect_T}, reçu={T}"
        if self.expect_F is not None:
            assert F == self.expect_F, f"F attendu={self.expect_F}, reçu={F}"

        t0, t1 = self.time_idx; w0, w1 = self.wave_idx; s0, s1 = self.spec_idx
        x_time = self._slice_domain(x, t0, t1)
        x_wave = self._slice_domain(x, w0, w1)
        x_spec = self._slice_domain(x, s0, s1)
        return self.core(x_time, x_wave, x_spec, stage)

# ==== Modèle (auto-config selon 1er batch normalisé) ====
# --- Détection auto T/F sur 1er batch ---
xb, yb = next(iter(train_dataloader))
xb = normalize_TF(xb)
T_detect, F_detect = xb.shape[1], xb.shape[2]
print(f"[DEBUG] First batch (normalized) -> T={T_detect}, F={F_detect}")

USE_DIFF = (F_detect >= 48)   # DIFF dispo si >=48 colonnes
if USE_DIFF:
    d_temp_in, d_wave_in, d_spec_in = 22, 16, 10   # 11+11, 8+8, 5+5
else:
    d_temp_in, d_wave_in, d_spec_in = 11, 8, 5     # RAW seul

# IMPORTANT: d_input=T_detect pour éviter la lazy-init
core = Transformer_4d(
    d_temp=d_temp_in, d_wave=d_wave_in, d_spec=d_spec_in,
    d_model=d_model, d_output=d_output, d_hidden=d_hidden,
    q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask,
    device=DEVICE, d_input=T_detect
).to(DEVICE)

net = DASDomainWrapper(
    core_3branch=core,
    expect_T=T_detect,
    expect_F=F_detect,
    use_diff=USE_DIFF
).to(DEVICE)

# Sanity-check immédiat des tailles de branches
with torch.no_grad():
    x_tmp = xb.to(DEVICE)
    def sl(x, a, b, off=24, use_diff=True):
        raw = x[:, :, a:b]
        if use_diff and x.size(-1) >= off+24:
            diff = x[:, :, off+a:off+b]
            return torch.cat([raw, diff], dim=-1)
        return raw
    x_time = sl(x_tmp, 0, 11, use_diff=USE_DIFF)
    x_wave = sl(x_tmp, 11, 19, use_diff=USE_DIFF)
    x_spec = sl(x_tmp, 19, 24, use_diff=USE_DIFF)
    print(f"[DEBUG] branch dims -> time={x_time.shape[-1]}, wave={x_wave.shape[-1]}, spec={x_spec.shape[-1]}")
    assert x_time.shape[-1] == d_temp_in
    assert x_wave.shape[-1] == d_wave_in
    assert x_spec.shape[-1] == d_spec_in

# ==== Pertes & Optimiseur ====
loss_function = Myloss()  # si utilisé ailleurs
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(net.parameters(), lr=LR) if optimizer_name=='Adagrad' else optim.Adam(net.parameters(), lr=LR)

# ==== Tracking métriques ====
correct_on_train = []
correct_on_val   = []
loss_list        = []

# ==== Fonctions d’évaluation ====
def evaluate_on(dataloader, flag='eval_set'):
    """Accuracy (%) sur un dataloader; passe x complet au wrapper."""
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x = normalize_TF(x).to(DEVICE)
            y = y.to(DEVICE)
            out = net(x, 'test')
            logits = unpack_logits(out)
            preds = logits.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y.long()).sum().item()
    acc = round(100.0 * correct / max(total, 1), 2)
    print(f'Accuracy on {flag}: {acc:.2f} %')
    return acc

def evaluate_and_plot_confusion_matrix(model, dataloader, device, class_labels):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = normalize_TF(x).to(device)
            y = y.to(device)
            out = model(x, 'test')
            logits = unpack_logits(out)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    C = confusion_matrix(all_labels, all_preds)
    df = pd.DataFrame(C)

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, fmt='g', annot=True, cmap='Reds',
                xticklabels=class_labels,
                yticklabels=class_labels,
                annot_kws={"size": 12})
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.xticks(rotation=0,  fontsize=12)
    plt.yticks(rotation=90, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(reslut_figure_path), exist_ok=True)
    plt.savefig(os.path.join(reslut_figure_path, "GTN_confusion_matrix.jpg"))
    plt.show(); plt.close()

    # Métriques dérivées
    print("\n===== Performance Metrics (Test) =====")
    acc = np.trace(C) / np.sum(C)
    print('Accuracy: %.4f' % acc)
    NAR = (np.sum(C[0]) - C[0][0]) / np.sum(C[:, 1:])
    print('NAR: %.4f' % NAR)
    FNR = (np.sum(C[:, 0]) - C[0][0]) / np.sum(C[1:])
    print('FNR: %.4f' % FNR)
    column_sum = np.sum(C, axis=0)
    row_sum = np.sum(C, axis=1)
    print('Column sums:', column_sum)
    print('Row sums:', row_sum)
    for i in range(len(class_labels)):
        TP = C[i][i]
        precision = TP / column_sum[i] if column_sum[i] != 0 else 0.0
        recall    = TP / row_sum[i]    if row_sum[i]    != 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
        print(f'Precision_{i}: {precision:.3f}')
        print(f'Recall_{i}:    {recall:.3f}')
        print(f'F1_{i}:        {f1:.3f}')

# ==== Entraînement (validation + logs gates/per-class + robustesse) ====
def train():
    print("\n📊 ==== Model Summary ====")
    count_parameters(net)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    best_val_acc   = -1.0
    best_model_tmp = os.path.join(save_dir, f'{file_name} batch={BATCH_SIZE}.pth')

    gate_logger = GateLogger()
    class_labels = ['background', 'digging', 'knocking', 'watering', 'shaking', 'walking']
    perclass_logger = ClassMetrics(class_labels)

    begin = time()
    for epoch in range(EPOCH):
        net.train()
        epoch_loss, batch_count = 0.0, 0

        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCH}", unit="batch") as tepoch:
            for x, y in tepoch:
                optimizer.zero_grad()
                x = normalize_TF(x).to(DEVICE)
                y = y.to(DEVICE)

                # DEBUG sur 1er batch
                if batch_count == 0:
                    print(">>> batch x shape (normalized):", tuple(x.shape))
                    raw_time  = x[:, :, 0:11]
                    diff_time = x[:, :, 24:35]
                    print(">>> raw_time", tuple(raw_time.shape), "diff_time", tuple(diff_time.shape))

                out = net(x, 'train')
                logits = unpack_logits(out)
                loss = criterion(logits, y.long())
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                # Monitoring (optionnel)
                try:
                    import psutil
                    ram_gb = psutil.virtual_memory().used / 1024**3
                    postfix = {"loss": f"{loss.item():.4f}", "RAM": f"{ram_gb:.2f}GB"}
                    if torch.cuda.is_available():
                        alloc  = torch.cuda.memory_allocated(DEVICE) / 1024**2
                        reserv = torch.cuda.memory_reserved(DEVICE) / 1024**2
                        postfix.update({"GPU_alloc": f"{alloc:.1f}MB", "GPU_resv": f"{reserv:.1f}MB"})
                    tepoch.set_postfix(postfix)
                except Exception:
                    pass

        mean_loss = epoch_loss / max(batch_count, 1)
        loss_list.append(mean_loss)
        print(f"Epoch {epoch+1}: Mean Loss = {mean_loss:.4f}")

        # === Validation à chaque epoch ===
        val_acc = evaluate_on(val_dataloader, flag='validation_set')
        train_acc = evaluate_on(train_dataloader, flag='train_set')
        correct_on_val.append(val_acc)
        correct_on_train.append(train_acc)

        # --- logging gates + per-class metrics sur val ---
        all_preds, all_trues = [], []
        with torch.no_grad():
            net.eval()
            first_batch_done = False
            for xb, yb in val_dataloader:
                xb = normalize_TF(xb).to(DEVICE)
                yb = yb.to(DEVICE)
                out = net(xb, 'test')
                logits = unpack_logits(out)
                preds = logits.argmax(dim=1)
                all_trues.append(yb.detach().cpu().numpy())
                all_preds.append(preds.detach().cpu().numpy())
                if not first_batch_done:
                    gate_logger.log_from_output_tuple(out)  # log des gates sur 1er batch val
                    first_batch_done = True
        y_true = np.concatenate(all_trues, axis=0)
        y_pred = np.concatenate(all_preds, axis=0)
        perclass_logger.update_from_preds(y_true, y_pred)

        # Sauvegarde si meilleur score val
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), best_model_tmp)
            print(f"💾 Saved new best (val) model @ {best_val_acc:.2f}% -> {best_model_tmp}")

        # Éval test périodique (facultatif)
        if (epoch + 1) % test_interval == 0:
            _ = evaluate_on(test_dataloader, flag='test_set (periodic)')

    # Renommage du meilleur modèle avec score val
    try:
        best_model_named = os.path.join(save_dir, f'{file_name} {best_val_acc:.2f}%_val batch={BATCH_SIZE}.pth')
        os.replace(best_model_tmp, best_model_named)
        best_model_path = best_model_named
    except Exception as e:
        print(f"[⚠️ rename failed] {e}")
        best_model_path = best_model_tmp

    end = time()
    time_cost = round((end - begin) / 60, 2)
    print(f"\n⏱️ Training+Validation completed in {time_cost} min.")

    # Recharger le meilleur modèle avant test final
    try:
        net.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print("✅ Loaded best (val) checkpoint for final testing.")
    except Exception as e:
        print(f"[⚠️ load best failed] {e}")

    # === TEST FINAL ===
    final_test_acc = evaluate_on(test_dataloader, flag='test_set (final)')
    print(f"🎯 Final Test Accuracy: {final_test_acc:.2f}%")

    # Matrice de confusion test
    evaluate_and_plot_confusion_matrix(
        model=net,
        dataloader=test_dataloader,
        device=DEVICE,
        class_labels=class_labels,
    )

    # Tracer l’évolution des gates (sur epochs)
    gate_fig = os.path.join(reslut_figure_path, "gate_evolution.png")
    gate_logger.plot(gate_fig)

    # Tracer les courbes per-class (F1/Precision/Recall)
    perclass_outdir = os.path.join(reslut_figure_path, "per_class_metrics")
    perclass_logger.plot(perclass_outdir)

    # Stress-test robustesse au bruit
    _ = evaluate_with_noise(net, test_dataloader, DEVICE,
                            std_list=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2))

    # Visualisation finale (ta fonction utilitaire)
    result_visualization(
        loss_list=loss_list,
        correct_on_test=correct_on_val,      # <- validation
        correct_on_train=correct_on_train,   # <- train
        test_interval=1,                     # val loggée à chaque epoch
        d_model=d_model, q=q, v=v, h=h, N=N,
        dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
        time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key,
        reslut_figure_path=reslut_figure_path, file_name=file_name,
        optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask
    )

# ==== Lancement ====
if __name__ == '__main__':
    train()
