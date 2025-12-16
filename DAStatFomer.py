# -*- coding: utf-8 -*-
"""
GTN 3 branches (time / waveform / spectral)
Entrée attendue: x (B, L, F) avec F=24
Le wrapper découpe:
  - time : 0:7 -> 7
  - wave : 7:13 -> 6
  - spec : 13:24 -> 11
Entraînement avec logs par EPOCH uniquement.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ====== Imports ======
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from pathlib import Path
from sklearn.metrics import confusion_matrix

# vos modules
from dataset_process.dataset_process import MyDataset
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization

try:
    from module.encoder import EncoderV2 as EncoderUsed
except Exception:
    from module.encoder import Encoder as EncoderUsed


# ====== GTN: 1 branche ======
class GTNBranch(nn.Module):
    """
    Une branche GTN:
      - embedding "step-wise": Linear(d_attr -> d_model) sur l'axe des features
      - embedding "channel-wise": Linear(L -> d_model) sur l'axe temporel
      - 2 piles d'encodeurs (step & chan)
      - gate 2-voies (step vs chan)
      - sortie vectorisée concat(step_flat, chan_flat) après pondération par la gate
    """
    def __init__(self, d_attr, d_input_L, d_model, d_hidden, q, v, h, N, dropout, device, pe=True, name="branch"):
        super().__init__()
        self.name = name
        self.L = d_input_L
        self.d_attr = d_attr
        self.d_model = d_model
        self.pe = pe

        self.embedding_step = nn.Linear(d_attr, d_model)     # (B,L,d_attr)->(B,L,d_model)
        self.embedding_chan = nn.Linear(self.L, d_model)      # (B,d_attr,L)->(B,d_attr,d_model)

        self.encoder_list_step = nn.ModuleList([
            EncoderUsed(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h, dropout=dropout, device=device)
            for _ in range(N)
        ])
        self.encoder_list_chan = nn.ModuleList([
            EncoderUsed(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h, dropout=dropout, device=device)
            for _ in range(N)
        ])

        fused_in_dim = d_model * self.L + d_model * d_attr
        self.gate = nn.Linear(fused_in_dim, 2)

    def _add_pe(self, x: torch.Tensor) -> torch.Tensor:
        if not self.pe:
            return x
        B, L, D = x.shape
        device = x.device
        pe = torch.zeros(L, D, device=device)
        pos = torch.arange(0, L, dtype=torch.float32, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return x + pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, stage: str):
        # x: (B, L, d_attr)
        # step-wise
        step_seq = self.embedding_step(x)
        step_seq = self._add_pe(step_seq)
        score_step = None
        for enc in self.encoder_list_step:
            step_seq, score_step = enc(step_seq, stage)   # (B,L,d_model)

        # channel-wise
        chan_seq = self.embedding_chan(x.transpose(-1, -2))   # (B,d_attr,d_model)
        score_chan = None
        for enc in self.encoder_list_chan:
            chan_seq, score_chan = enc(chan_seq, stage)

        # flatten + gate
        step_flat = step_seq.reshape(step_seq.size(0), -1)
        chan_flat = chan_seq.reshape(chan_seq.size(0), -1)
        gate = F.softmax(self.gate(torch.cat([step_flat, chan_flat], dim=-1)), dim=-1)  # (B,2)

        fused_branch = torch.cat([step_flat * gate[:, 0:1], chan_flat * gate[:, 1:2]], dim=-1)
        return fused_branch, gate, step_seq, chan_seq, score_step, score_chan


# ====== GTN: 3 branches parallèles + fusion ======
class GTNParallel3(nn.Module):
    """
    forward(x_time, x_wave, x_spec, stage)
      x_* : (B, L, d_attr_*)
    Sortie: (logits, fused_all, scores_step_list, scores_chan_list, seqs_step_list, seqs_chan_list, gates_list)
    """
    def __init__(self, d_temp, d_wave, d_spec, d_input_L,
                 d_model, d_hidden, q, v, h, N, dropout, device,
                 d_output, pe=True):
        super().__init__()
        self.d_output = d_output

        self.branch_time = GTNBranch(d_attr=d_temp, d_input_L=d_input_L,
                                     d_model=d_model, d_hidden=d_hidden,
                                     q=q, v=v, h=h, N=N, dropout=dropout,
                                     device=device, pe=pe, name="time")

        self.branch_wave = GTNBranch(d_attr=d_wave, d_input_L=d_input_L,
                                     d_model=d_model, d_hidden=d_hidden,
                                     q=q, v=v, h=h, N=N, dropout=dropout,
                                     device=device, pe=pe, name="wave")

        self.branch_spec = GTNBranch(d_attr=d_spec, d_input_L=d_input_L,
                                     d_model=d_model, d_hidden=d_hidden,
                                     q=q, v=v, h=h, N=N, dropout=dropout,
                                     device=device, pe=pe, name="spec")

        fused_dim = d_model * (d_input_L + d_temp) \
                  + d_model * (d_input_L + d_wave) \
                  + d_model * (d_input_L + d_spec)
        self.output_layer = nn.Linear(fused_dim, d_output)

    def forward(self, x_time, x_wave, x_spec, stage: str):
        t_fused, t_gate, t_step, t_chan, t_s_step, t_s_chan = self.branch_time(x_time, stage)
        w_fused, w_gate, w_step, w_chan, w_s_step, w_s_chan = self.branch_wave(x_wave, stage)
        s_fused, s_gate, s_step, s_chan, s_s_step, s_s_chan = self.branch_spec(x_spec, stage)

        fused_all = torch.cat([t_fused, w_fused, s_fused], dim=-1)
        logits = self.output_layer(fused_all)
        gates_list = [t_gate, w_gate, s_gate]
        scores_step_list = [t_s_step, w_s_step, s_s_step]
        scores_chan_list = [t_s_chan, w_s_chan, s_s_chan]
        seqs_step_list   = [t_step,   w_step,   s_step]
        seqs_chan_list   = [t_chan,   w_chan,   s_chan]
        return (logits, fused_all, scores_step_list, scores_chan_list,
                seqs_step_list, seqs_chan_list, gates_list)


# ====== Wrapper : découpe x_full (B,L,F) -> 3 domaines RAW+DIFF ======
class DASDomainWrapper(nn.Module):
    """
    Par défaut (RAW+DIFF, F=48):
      time: 0:11 + 24:35 (22), waveform: 11:19 + 35:43 (16), spectral: 19:24 + 43:48 (10)
    """
    def __init__(self, core_3branch: GTNParallel3,
                 time_idx=(0,7), wave_idx=(7,13), spec_idx=(13,24),
                 diff_offset=24, use_diff=True):
        super().__init__()
        self.core = core_3branch
        self.t0, self.t1 = time_idx
        self.w0, self.w1 = wave_idx
        self.s0, self.s1 = spec_idx
        self.diff_offset = diff_offset
        self.use_diff = use_diff

    def _slice(self, x, a, b):
        raw = x[:, :, a:b]
        if self.use_diff:
            diff = x[:, :, self.diff_offset + a : self.diff_offset + b]
            return torch.cat([raw, diff], dim=-1)
        return raw

    def forward(self, x_full, stage: str):
        # On veut toujours (B, L, F) avec F = nb de features (time/wave/spec)
        # Si la dernière dim est plus petite que la précédente, on transpose
        if x_full.dim() == 3 and x_full.shape[2] < x_full.shape[1]:
            x_full = x_full.transpose(1, 2)   # ex: (B,24,12) -> (B,12,24)
    
        x_time = self._slice(x_full, self.t0, self.t1)
        x_wave = self._slice(x_full, self.w0, self.w1)
        x_spec = self._slice(x_full, self.s0, self.s1)
    
        return self.core(x_time, x_wave, x_spec, stage)



# ====== Plots ======
def draw_loss_acc(train_acc, train_loss, val_acc, val_loss, out_png: str) -> None:
    import numpy as np
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    n = min(len(train_acc), len(val_acc), len(train_loss), len(val_loss))
    train_acc, val_acc   = list(train_acc[:n]), list(val_acc[:n])
    train_loss, val_loss = list(train_loss[:n]), list(val_loss[:n])
    epochs = range(n)

    plt.figure(figsize=(8, 8))
    # ACC
    plt.subplot(2,1,1)
    plt.plot(epochs, train_acc, label="train", linewidth=1.5)
    plt.plot(epochs, val_acc,   label="val",   linewidth=1.5)
    plt.title('Accuracy vs. epochs'); plt.ylabel('Accuracy'); plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3); plt.legend(loc='upper left')
    # LOSS
    plt.subplot(2,1,2)
    plt.plot(epochs, train_loss, label="train", linewidth=1.5)
    plt.plot(epochs, val_loss,   label="val",   linewidth=1.5)
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    ymax = max(1.0, max(train_loss + val_loss)) * 1.05
    plt.ylim(0.0, ymax); plt.grid(True, alpha=0.3); plt.legend(loc='upper left')
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.show(); plt.close()


def evaluate_and_plot_confusion_matrix(model, dataloader, device, class_labels, save_path, file_name):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, *_ = model(x, 'test')
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = y.cpu().numpy()
            all_preds.extend(preds); all_labels.extend(labels)

    C = confusion_matrix(all_labels, all_preds)
    df = pd.DataFrame(C)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, fmt='g', annot=True, cmap='Reds',
                xticklabels=class_labels, yticklabels=class_labels,
                annot_kws={"size": 12})
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(rotation=90, fontsize=12)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{file_name}_Conf_Max.png')
    plt.show(); plt.close()

    print("\n===== Performance Metrics =====")
    acc = np.trace(C) / np.sum(C); print('Accuracy: %.4f' % acc)
    NAR = (np.sum(C[0]) - C[0][0]) / np.sum(C[:, 1:]) if C.shape[1] > 1 else 0.0
    print('NAR: %.4f' % NAR)
    FNR = (np.sum(C[:, 0]) - C[0][0]) / np.sum(C[1:]) if C.shape[0] > 1 else 0.0
    print('FNR: %.4f' % FNR)
    column_sum = np.sum(C, axis=0); row_sum = np.sum(C, axis=1)
    print('Column sums:', column_sum); print('Row sums:', row_sum)
    for i in range(len(class_labels)):
        TP = C[i][i]
        precision = TP / column_sum[i] if column_sum[i] != 0 else 0.0
        recall = TP / row_sum[i] if row_sum[i] != 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
        print(f'Precision_{i}: {precision:.3f}')
        print(f'Recall_{i}:    {recall:.3f}')
        print(f'F1_{i}:        {f1:.3f}')


def evaluate_loss_acc(model, dataloader, device, criterion, desc="valid"):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, *_ = model(x, 'test')
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_count += y.size(0)
    mean_loss = total_loss / max(total_count, 1)
    acc = 100.0 * total_correct / max(total_count, 1)
    print(f"{desc}: loss={mean_loss:.4f} | acc={acc:.2f}%")
    return mean_loss, acc


# ====== Main config ======
setup_seed(30)
reslut_figure_path = r'D:\Michel\DAStatFormer\results_figure'
path_mat = r'D:\Michel\Gated Transformer 论文IJCAI版\DAS_DataNorm_24_features_selected.mat' #r'D:\Michel\DAStatFormer\DAS_DataNorm_all_domains_for_GTN.mat'
test_interval = 5
draw_key = 1
file_name = Path(path_mat).name.split('.')[0]

EPOCH = 100
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {DEVICE}')

d_model = 512
d_hidden = 128
q = 8; v = 8; h = 8
N = 8
dropout = 0.2
pe = True
mask = True  # (utilisé dans les Encoders)
optimizer_name = 'Adam'

# # ====== Data ======
# train_dataset = MyDataset(path_mat, 'train')
# test_dataset  = MyDataset(path_mat, 'test')
# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# DATA_LEN = train_dataset.train_len
# d_input  = train_dataset.input_len   # L
# d_channel = train_dataset.channel_len # F
# d_output = train_dataset.output_len


from torch.utils.data import random_split, DataLoader


base_train_ds = MyDataset(path_mat, 'train')
test_dataset  = MyDataset(path_mat, 'test')


n_total = len(base_train_ds)
n_val   = int(0.1 * n_total)
n_train = n_total - n_val

train_dataset, val_dataset = random_split(
    base_train_ds,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

# 3) DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# 4) Métadonnées (prendre sur le dataset de base, pas sur le Subset)
DATA_LEN  = len(train_dataset)                # au lieu de base_train_ds.train_len
d_input   = base_train_ds.input_len
d_channel = base_train_ds.channel_len
d_output  = base_train_ds.output_len


# ====== Déduction L,F & choix RAW+DIFF ======
xb, yb = next(iter(train_dataloader))

if xb.shape[2] < xb.shape[1]:
    
    xb = xb.transpose(1, 2)

L_detect, F_detect = xb.shape[1], xb.shape[2]
USE_DIFF = (F_detect >= 48)

print(f"Using normalized shape: L={L_detect}, F={F_detect}")


if USE_DIFF:
    time_idx = (0, 14)
    wave_idx = (14, 26)
    spec_idx = (26, 48)
else:
    # 24 features = 7 (temp) + 6 (wave) + 11 (spec)
    time_idx = (0, 7)
    wave_idx = (7, 13)
    spec_idx = (13, 24)

t0, t1 = time_idx
w0, w1 = wave_idx
s0, s1 = spec_idx

d_temp_in = t1 - t0
d_wave_in = w1 - w0
d_spec_in = s1 - s0


core = GTNParallel3(
    d_temp=d_temp_in, d_wave=d_wave_in, d_spec=d_spec_in,
    d_input_L=L_detect,
    d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h, N=N,
    dropout=dropout, device=DEVICE, d_output=d_output, pe=pe
).to(DEVICE)

net = DASDomainWrapper(
    core_3branch=core,
    time_idx=time_idx, wave_idx=wave_idx, spec_idx=spec_idx,
    diff_offset=24, use_diff=USE_DIFF
).to(DEVICE)

print(f"[Model] L={L_detect}, F={F_detect}, use_diff={USE_DIFF} "
      f"-> d_temp={d_temp_in}, d_wave={d_wave_in}, d_spec={d_spec_in}")
with torch.no_grad():
    xb, yb = next(iter(train_dataloader))
    logits, *_ = net(xb.to(DEVICE), 'test')
    print("OK forward:", tuple(logits.shape))  # (B, num_classes)

# ====== Loss & Optim ======
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR) if optimizer_name == 'Adam' else optim.Adagrad(net.parameters(), lr=LR)

# ====== Suivi ======
correct_on_train, correct_on_test, loss_list = [], [], []

def test(dataloader, flag='test_set'):
    correct = 0; total = 0
    net.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, *_ = net(x, 'test')
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    acc = round(100.0 * correct / max(total, 1), 2)
    if flag == 'test_set': correct_on_test.append(acc)
    elif flag == 'train_set': correct_on_train.append(acc)
    print(f'Accuracy on {flag}: {acc:.2f} %')
    return acc

# ====== Train (progress par EPOCH seulement) ======
def train():
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("\n📊 ==== Model Summary ====\n")
    print(f"🔍 Total parameters       : {total_params:,}")
    print(f"🧠 Trainable parameters   : {trainable_params:,}")
    print(f"💾 Estimated model size   : {trainable_params * 4 / 1024 ** 2:.2f} MB (float32)\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    net.train()
    max_accuracy = 0.0
    begin = time()

    # buffers pour courbes
    train_losses_plot, val_losses_plot = [], []
    train_acc_plot, val_acc_plot = [], []

    for epoch in range(EPOCH):
        net.train()
        epoch_loss, batch_count = 0.0, 0

        # --- boucle TRAIN sans tqdm par batch ---
        for x, y in train_dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits, *_ = net(x, 'train')
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        mean_loss = epoch_loss / max(batch_count, 1)
        loss_list.append(mean_loss)
        print(f"Epoch {epoch+1}/{EPOCH} | Train Loss = {mean_loss:.4f}")
        
        
        tr_loss_ep, tr_acc_ep = evaluate_loss_acc(net, train_dataloader, DEVICE, criterion, desc="train")
        va_loss_ep, va_acc_ep = evaluate_loss_acc(net, val_dataloader, DEVICE, criterion, desc="valid")
        # ...
        if (epoch + 1) % test_interval == 0:
            current_accuracy = va_acc_ep   # validation accuracy, pas test
            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                os.makedirs('D:/Michel/DAStatFormer/saved_model_3para', exist_ok=True)
                torch.save(net, f'D:/Michel/DAStatFormer/saved_model/{file_name} batch={BATCH_SIZE}.pkl')

        # # Évaluations rapides (pour courbes)
        # tr_loss_ep, tr_acc_ep = evaluate_loss_acc(net, train_dataloader, DEVICE, criterion, desc="train")
        # va_loss_ep, va_acc_ep = evaluate_loss_acc(net, test_dataloader,  DEVICE, criterion, desc="valid")
        # train_losses_plot.append(tr_loss_ep); val_losses_plot.append(va_loss_ep)
        # train_acc_plot.append(tr_acc_ep/100.0); val_acc_plot.append(va_acc_ep/100.0)

        # # test périodique
        # if (epoch + 1) % test_interval == 0:
        #     current_accuracy = test(test_dataloader, 'test_set')
        #     test(train_dataloader, 'train_set')
        #     print(f"Max Accuracy so far — Test: {max(correct_on_test)}% | Train: {max(correct_on_train)}%")
        #     if current_accuracy > max_accuracy:
        #         max_accuracy = current_accuracy
        #         os.makedirs('D:/Michel/DAStatFormer/saved_model', exist_ok=True)
        #         torch.save(net, f'D:/Michel/DAStatFormer/saved_model/{file_name} batch={BATCH_SIZE}.pkl')

    # rename best
    try:
        os.replace(f'D:/Michel/DAStatFormer/saved_model/{file_name} batch={BATCH_SIZE}.pkl',
                   f'D:/Michel/DAStatFormer/saved_model/{file_name} {max_accuracy:.2f}% batch={BATCH_SIZE}.pkl')
    except Exception as e:
        print(f"[rename failed] {e}")

    time_cost = round((time() - begin) / 60, 2)
    print(f"\n⏱️ Training completed in {time_cost} min.")

    if torch.cuda.is_available():
        max_alloc = torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2
        max_reserv = torch.cuda.max_memory_reserved(DEVICE) / 1024 ** 2
        print(f"📈 Max GPU Memory Allocated: {max_alloc:.2f} MB")
        print(f"📦 Max GPU Memory Reserved : {max_reserv:.2f} MB")

    # Courbes finales
    try:
        final_png = os.path.join(reslut_figure_path, f"{file_name}_loss_acc.png")
        draw_loss_acc(train_acc_plot, train_losses_plot, val_acc_plot, val_losses_plot, final_png)
        print(f"[OK] Courbes loss/acc sauvegardées : {final_png}")
    except Exception as e:
        print(f"[final plot failed] {e}")

    # Confusion + métriques

    evaluate_and_plot_confusion_matrix(
        model=net,
        dataloader=test_dataloader,
        device=DEVICE,
        class_labels=['background', 'digging', 'knocking', 'watering', 'shaking', 'walking'],save_path=reslut_figure_path, file_name=file_name,
    )
    
    avg_infer_time_ms = measure_inference_time(net, test_dataloader, DEVICE)
    print(f"Average inference time per sample: {avg_infer_time_ms:.3f} ms")

    # Visualisation (votre utilitaire)
    result_visualization(
        loss_list=loss_list,
        correct_on_test=correct_on_test,
        correct_on_train=correct_on_train,
        test_interval=test_interval,
        d_model=d_model, q=q, v=v, h=h, N=N,
        dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
        time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key,
        reslut_figure_path=reslut_figure_path, file_name=file_name,
        optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask
    )

import time as tm

def measure_inference_time(model, dataloader, device, n_warmup=5):
    """
    Measure the average inference time per sample (in milliseconds).
    - n_warmup: nombre d’itérations ignorées pour stabiliser le GPU.
    """
    model.eval()
    times = []
    total_samples = 0

    with torch.no_grad():
        # Boucle principale
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            batch_size = x.size(0)

            
            if i < n_warmup:
                _ = model(x, 'test')
                continue

            start_time = tm.perf_counter()
            _ = model(x, 'test')
            end_time = tm.perf_counter()

            elapsed = end_time - start_time
            times.append(elapsed)
            total_samples += batch_size

    
    total_time = np.sum(times)
    avg_time_per_batch = total_time / len(times)
    avg_time_per_sample = (total_time / total_samples) * 1000  # en ms

    print(f"\n⚡ Inference time per batch:   {avg_time_per_batch:.4f} s")
    print(f"⚡ Inference time per sample: {avg_time_per_sample:.4f} ms\n")

    return avg_time_per_sample


if __name__ == '__main__':
    train()
