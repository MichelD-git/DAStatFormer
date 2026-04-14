# -*- coding: utf-8 -*-

"""
Created on Fri Oct 10 11:11:09 2025

@author: michel
"""

"""
GTN 3 branches (time / waveform / spectral)
Input x (B, L, F) avec F=24
wrapper :
  - time : 0:7 -> 7
  - wave : 7:13 -> 6
  - spec : 13:24 -> 11
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import time as tm
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, random_split, Dataset

from dataset_process.dataset_process import MyDataset
from utils.random_seed import setup_seed
from utils.visualization import result_visualization
from module.feedForward import FeedForward
from module.encoder import Encoder


# =========================================================
# MultiHeadAttention
# =========================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, q: int, v: int, h: int, device: str,
                 mask: bool = False, dropout: float = 0.1):
        super().__init__()
        self.W_q = nn.Linear(d_model, q * h)
        self.W_k = nn.Linear(d_model, q * h)
        self.W_v = nn.Linear(d_model, v * h)
        self.W_o = nn.Linear(v * h, d_model)

        self.device = device
        self._h = h
        self._q = q
        self.mask = mask
        self.dropout = nn.Dropout(p=dropout)
        self.score = None

    def forward(self, q_in, kv_in=None, stage='train'):
        if kv_in is None:
            kv_in = q_in

        Q = torch.cat(self.W_q(q_in).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.W_k(kv_in).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.W_v(kv_in).chunk(self._h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score

        if self.mask and (kv_in is q_in) and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(
                mask > 0, score,
                torch.tensor([-2**32 + 1], device=self.device).expand_as(score[0])
            )

        score = F.softmax(score, dim=-1)
        score = self.dropout(score)

        attention = torch.matmul(score, V)
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)
        out = self.W_o(attention_heads)

        return out, self.score


# =========================================================
# CrossEncoder
# =========================================================
class CrossEncoder(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, q: int, v: int, h: int,
                 device: str, dropout: float = 0.1):
        super().__init__()

        self.cross_mha = MultiHeadAttention(
            d_model=d_model,
            q=q,
            v=v,
            h=h,
            mask=False,
            device=device,
            dropout=dropout
        )

        self.feedforward = FeedForward(
            d_model=d_model,
            d_hidden=d_hidden
        )

        self.dropout = nn.Dropout(p=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, q_x, kv_x, stage):
        residual = q_x
        q_x_norm = self.ln1(q_x)
        out, score = self.cross_mha(q_x_norm, kv_x, stage=stage)
        q_x = residual + self.dropout(out)

        residual = q_x
        q_x_norm = self.ln2(q_x)
        out = self.feedforward(q_x_norm)
        q_x = residual + self.dropout(out)

        return q_x, score


# =========================================================
# Dataset wrapper : split des 24 features
# =========================================================
class MyDataset3Domains(Dataset):
    def __init__(self, path, split):
        self.base = MyDataset(path, split)

        self.input_len = self.base.input_len
        self.output_len = self.base.output_len

        self.time_dim = 7
        self.wave_dim = 6
        self.spec_dim = 11

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]

        # On force tensor float
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()

        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = y.long()

        if x.ndim != 2:
            raise ValueError(f"Entrée attendue 2D, reçu {tuple(x.shape)}")

        # Cas 1 : x = (L, 24)
        if x.shape[1] == 24:
            x_lf = x

        # Cas 2 : x = (24, L)  --> on transpose
        elif x.shape[0] == 24:
            x_lf = x.transpose(0, 1)

        else:
            raise ValueError(
                f"Impossible de trouver l'axe des 24 features. "
                f"Shape reçue : {tuple(x.shape)}"
            )

        # x_lf = (L, 24)
        x_time = x_lf[:, 0:7]
        x_wave = x_lf[:, 7:13]
        x_spec = x_lf[:, 13:24]

        return x_time, x_wave, x_spec, y


# =========================================================
# Une branche GTN
# =========================================================
class GTNBranch(nn.Module):
    def __init__(self, d_attr, d_input_L, d_model, d_hidden, q, v, h, N,
                 dropout, device, pe=True, mask=False, name="branch"):
        super().__init__()
        self.name = name
        self.L = d_input_L
        self.d_attr = d_attr
        self.d_model = d_model
        self.pe = pe

        # step-wise: (B, L, d_attr) -> (B, L, d_model)
        self.embedding_step = nn.Linear(d_attr, d_model)

        # channel-wise/domain-wise: (B, d_attr, L) -> (B, d_attr, d_model)
        self.embedding_chan = nn.Linear(self.L, d_model)

        self.encoder_list_step = nn.ModuleList([
            Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
                    mask=mask, dropout=dropout, device=device)
            for _ in range(N)
        ])

        self.encoder_list_chan = nn.ModuleList([
            Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
                    mask=False, dropout=dropout, device=device)
            for _ in range(N)
        ])

        self.cross_step_to_chan = CrossEncoder(
            d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
            device=device, dropout=dropout
        )
        self.cross_chan_to_step = CrossEncoder(
            d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h,
            device=device, dropout=dropout
        )

        fused_in_dim = d_model * self.L + d_model * d_attr
        self.gate = nn.Linear(fused_in_dim, 2)

    def _add_pe(self, x):
        if not self.pe:
            return x

        _, L, D = x.shape
        device = x.device
        pe = torch.zeros(L, D, device=device)
        pos = torch.arange(0, L, dtype=torch.float32, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return x + pe.unsqueeze(0)

    def forward(self, x, stage):
        # x : (B, L, d_attr)

        # -------- Step-wise --------
        step_seq = self.embedding_step(x)
        step_seq = self._add_pe(step_seq)

        score_step = None
        for enc in self.encoder_list_step:
            step_seq, score_step = enc(step_seq, stage)

        # -------- Channel/domain-wise --------
        chan_seq = self.embedding_chan(x.transpose(-1, -2))  # (B, d_attr, L) -> (B, d_attr, d_model)

        score_chan = None
        for enc in self.encoder_list_chan:
            chan_seq, score_chan = enc(chan_seq, stage)

        # -------- Cross-attention bidirectionnelle --------
        step_seq, score_cross_s2c = self.cross_step_to_chan(step_seq, chan_seq, stage)
        chan_seq, score_cross_c2s = self.cross_chan_to_step(chan_seq, step_seq, stage)

        # -------- Flatten + gate --------
        step_flat = step_seq.reshape(step_seq.size(0), -1)
        chan_flat = chan_seq.reshape(chan_seq.size(0), -1)

        gate = F.softmax(self.gate(torch.cat([step_flat, chan_flat], dim=-1)), dim=-1)

        fused_branch = torch.cat([
            step_flat * gate[:, 0:1],
            chan_flat * gate[:, 1:2]
        ], dim=-1)

        return (
            fused_branch, gate,
            step_seq, chan_seq,
            score_step, score_chan,
            score_cross_s2c, score_cross_c2s
        )


# =========================================================
# Modèle 3 branches
# =========================================================
class GTNParallel3Domains(nn.Module):
    def __init__(self, d_temp, d_wave, d_spec, d_input_L,
                 d_model, d_hidden, q, v, h, N, dropout, device,
                 d_output, pe=True, mask=False):
        super().__init__()

        self.branch_time = GTNBranch(
            d_attr=d_temp,
            d_input_L=d_input_L,
            d_model=d_model,
            d_hidden=d_hidden,
            q=q, v=v, h=h, N=N,
            dropout=dropout,
            device=device,
            pe=pe,
            mask=mask,
            name="time"
        )

        self.branch_wave = GTNBranch(
            d_attr=d_wave,
            d_input_L=d_input_L,
            d_model=d_model,
            d_hidden=d_hidden,
            q=q, v=v, h=h, N=N,
            dropout=dropout,
            device=device,
            pe=pe,
            mask=mask,
            name="wave"
        )

        self.branch_spec = GTNBranch(
            d_attr=d_spec,
            d_input_L=d_input_L,
            d_model=d_model,
            d_hidden=d_hidden,
            q=q, v=v, h=h, N=N,
            dropout=dropout,
            device=device,
            pe=pe,
            mask=mask,
            name="spec"
        )

        fused_dim = (
            d_model * (d_input_L + d_temp) +
            d_model * (d_input_L + d_wave) +
            d_model * (d_input_L + d_spec)
        )

        self.output_layer = nn.Linear(fused_dim, d_output)

    def forward(self, x_time, x_wave, x_spec, stage):
        out_t = self.branch_time(x_time, stage)
        out_w = self.branch_wave(x_wave, stage)
        out_s = self.branch_spec(x_spec, stage)

        t_fused, t_gate = out_t[0], out_t[1]
        w_fused, w_gate = out_w[0], out_w[1]
        s_fused, s_gate = out_s[0], out_s[1]

        fused_all = torch.cat([t_fused, w_fused, s_fused], dim=-1)
        logits = self.output_layer(fused_all)

        return logits, fused_all, out_t, out_w, out_s


# =========================================================
# Eval functions
# =========================================================
def evaluate_loss_acc(model, dataloader, device, criterion, desc="valid"):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0

    with torch.no_grad():
        for x_time, x_wave, x_spec, y in dataloader:
            x_time = x_time.to(device)
            x_wave = x_wave.to(device)
            x_spec = x_spec.to(device)
            y = y.to(device)

            logits, *_ = model(x_time, x_wave, x_spec, 'test')
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_count += y.size(0)

    mean_loss = total_loss / max(total_count, 1)
    acc = 100.0 * total_correct / max(total_count, 1)
    print(f"{desc}: loss={mean_loss:.4f} | acc={acc:.2f}%")
    return mean_loss, acc


def evaluate_and_plot_confusion_matrix(model, dataloader, device, class_labels, save_path, file_name,
                                       background_index=0):
    all_preds, all_labels = [], []
    model.eval()

    with torch.no_grad():
        for x_time, x_wave, x_spec, y in dataloader:
            x_time = x_time.to(device)
            x_wave = x_wave.to(device)
            x_spec = x_spec.to(device)
            y = y.to(device)

            y_pre, *_ = model(x_time, x_wave, x_spec, 'test')
            preds = y_pre.argmax(dim=1).cpu().numpy()
            labels = y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    C = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_labels))))
    df = pd.DataFrame(C, index=class_labels, columns=class_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, fmt='g', annot=True, cmap='Reds',
                xticklabels=class_labels,
                yticklabels=class_labels,
                annot_kws={"size": 12})
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(rotation=90, fontsize=12)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{file_name}_Conf_Max.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print("\n===== Performance Metrics =====")
    acc = np.trace(C) / np.sum(C) if np.sum(C) != 0 else 0.0
    print(f'Accuracy: {acc:.4f}')

    bg = background_index
    if C.shape[0] > 1:
        nar_den = np.sum(C[:, [j for j in range(C.shape[1]) if j != bg]])
        nar = (np.sum(C[bg, :]) - C[bg, bg]) / nar_den if nar_den != 0 else 0.0

        fnr_den = np.sum(C[[i for i in range(C.shape[0]) if i != bg], :])
        fnr = (np.sum(C[:, bg]) - C[bg, bg]) / fnr_den if fnr_den != 0 else 0.0

        print(f'NAR: {nar:.4f}')
        print(f'FNR: {fnr:.4f}')

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    print(f'Macro Precision: {precision_macro:.4f}')
    print(f'Macro Recall:    {recall_macro:.4f}')
    print(f'Macro F1-score:  {f1_macro:.4f}')

    column_sum = np.sum(C, axis=0)
    row_sum = np.sum(C, axis=1)

    for i in range(len(class_labels)):
        TP = C[i, i]
        precision = TP / column_sum[i] if column_sum[i] != 0 else 0.0
        recall = TP / row_sum[i] if row_sum[i] != 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0

        print(f'\nClass {class_labels[i]}')
        print(f'Precision: {precision:.3f}')
        print(f'Recall:    {recall:.3f}')
        print(f'F1-score:  {f1:.3f}')


def measure_inference_time(model, dataloader, device, n_warmup=5):
    model.eval()
    times = []
    total_samples = 0

    with torch.no_grad():
        for i, (x_time, x_wave, x_spec, _) in enumerate(dataloader):
            x_time = x_time.to(device)
            x_wave = x_wave.to(device)
            x_spec = x_spec.to(device)
            bs = x_time.size(0)

            if i < n_warmup:
                _ = model(x_time, x_wave, x_spec, 'test')
                continue

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = tm.perf_counter()

            _ = model(x_time, x_wave, x_spec, 'test')

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = tm.perf_counter()

            times.append(t1 - t0)
            total_samples += bs

    total_time = float(np.sum(times)) if len(times) else 0.0
    avg_time_per_sample = (total_time / max(total_samples, 1)) * 1000.0
    print(f"\n⚡ Inference time per sample: {avg_time_per_sample:.4f} ms\n")
    return avg_time_per_sample


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔍 Total parameters       : {total_params:,}")
    print(f"🧠 Trainable parameters   : {trainable_params:,}")
    print(f"💾 Estimated model size   : {trainable_params * 4 / 1024 ** 2:.2f} MB (float32)\n")


# =========================================================
# Train
# =========================================================
def train(net, train_dataloader, val_dataloader, test_dataloader,
          DEVICE, EPOCH, test_interval, BATCH_SIZE,
          reslut_figure_path, save_path, file_name,
          criterion, optimizer,
          d_model, q, v, h, N, dropout, DATA_LEN, draw_key,
          optimizer_name, LR, pe, mask):

    print("\n📊 ==== Model Summary ====")
    count_parameters(net)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    begin = time()
    max_accuracy = -1.0

    train_losses_plot, val_losses_plot = [], []
    train_acc_plot, val_acc_plot = [], []

    correct_on_train, correct_on_test = [], []
    loss_list = []

    best_model_dir = r'D:\Michel\DAStatFormer_3domains_mat\saved_model'
    os.makedirs(best_model_dir, exist_ok=True)

    for epoch in range(EPOCH):
        net.train()
        epoch_loss = 0.0
        batch_count = 0

        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCH}", unit="batch") as tepoch:
            for x_time, x_wave, x_spec, y in tepoch:
                x_time = x_time.to(DEVICE)
                x_wave = x_wave.to(DEVICE)
                x_spec = x_spec.to(DEVICE)
                y = y.to(DEVICE)

                optimizer.zero_grad()
                y_pre, *_ = net(x_time, x_wave, x_spec, 'train')
                loss = criterion(y_pre, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                tepoch.set_postfix({"loss": f"{loss.item():.4f}"})

        mean_loss = epoch_loss / max(batch_count, 1)
        loss_list.append(mean_loss)
        print(f"Epoch {epoch+1}: Mean Loss = {mean_loss:.4f}")

        tr_loss_ep, tr_acc_ep = evaluate_loss_acc(net, train_dataloader, DEVICE, criterion, desc="train")
        va_loss_ep, va_acc_ep = evaluate_loss_acc(net, val_dataloader, DEVICE, criterion, desc="valid")

        train_losses_plot.append(tr_loss_ep)
        val_losses_plot.append(va_loss_ep)
        train_acc_plot.append(tr_acc_ep / 100.0)
        val_acc_plot.append(va_acc_ep / 100.0)

        correct_on_train.append(tr_acc_ep)
        correct_on_test.append(va_acc_ep)

        if (epoch + 1) % test_interval == 0:
            if va_acc_ep > max_accuracy:
                max_accuracy = va_acc_ep
                torch.save(net.state_dict(), os.path.join(best_model_dir, f'{file_name}_best.pkl'))

    try:
        src = os.path.join(best_model_dir, f'{file_name}_best.pkl')
        dst = os.path.join(best_model_dir, f'{file_name}_best_{max_accuracy:.2f}%_batch={BATCH_SIZE}.pkl')
        if os.path.exists(src):
            os.rename(src, dst)
            print(f"[OK] Renamed best model -> {dst}")
        else:
            print(f"[⚠️ rename skipped] Source not found: {src}")
    except Exception as e:
        print(f"[⚠️ os.rename failed] {e}")

    end = time()
    time_cost = round((end - begin) / 60, 2)
    print(f"\n⏱️ Training completed in {time_cost} min.")

    if torch.cuda.is_available():
        max_alloc = torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2
        max_reserv = torch.cuda.max_memory_reserved(DEVICE) / 1024 ** 2
        print(f"📈 Max GPU Memory Allocated: {max_alloc:.2f} MB")
        print(f"📦 Max GPU Memory Reserved : {max_reserv:.2f} MB")

    evaluate_and_plot_confusion_matrix(
        model=net,
        dataloader=test_dataloader,
        device=DEVICE,
        class_labels=[str(i) for i in range(net.output_layer.out_features)],
        save_path=save_path,
        file_name=file_name,
        background_index=0
    )

    avg_infer_time_ms = measure_inference_time(net, test_dataloader, DEVICE)
    print(f"Average inference time per sample: {avg_infer_time_ms:.3f} ms")

    result_visualization(
        loss_list_train=train_losses_plot,
        loss_list_val=val_losses_plot,
        correct_on_test=correct_on_test,
        correct_on_train=correct_on_train,
        test_interval=test_interval,
        d_model=d_model, q=q, v=v, h=h, N=N,
        dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
        time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key,
        reslut_figure_path=reslut_figure_path, file_name=file_name,
        optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask
    )


# =========================================================
# Launch
# =========================================================
if __name__ == '__main__':
    setup_seed(30)

    reslut_figure_path = r'D:\Michel\DAStatFormer_3domains_mat\results_figure'
    save_path = r'D:\Michel\DAStatFormer_3domains_mat\results_confusion_matrix'

    path = r'D:\Michel\Gated Transformer 论文IJCAI版\DAS_DataNorm_24_features_selected.mat'
    file_name = path.split('\\')[-1].split('.')[0]

    EPOCH = 100
    BATCH_SIZE = 32
    LR = 1e-4
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {DEVICE}')

    # Je conseille de commencer plus petit que ton ancien script
    d_model = 256
    d_hidden = 128
    q = 8
    v = 8
    h = 8
    N = 8
    dropout = 0.2
    pe = True
    mask = True
    optimizer_name = 'AdamW'
    test_interval = 5
    draw_key = 1

    base_train_ds = MyDataset3Domains(path, 'train')
    test_dataset = MyDataset3Domains(path, 'test')

    n_total = len(base_train_ds)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        base_train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    DATA_LEN = len(train_dataset)
    
    x_time0, x_wave0, x_spec0, y0 = base_train_ds[0]
    
    d_input = x_time0.shape[0]   # longueur réelle L
    d_output = base_train_ds.output_len
    
    d_temp = x_time0.shape[1]
    d_wave = x_wave0.shape[1]
    d_spec = x_spec0.shape[1]

    print(f"[Data] L={d_input}, d_temp={d_temp}, d_wave={d_wave}, d_spec={d_spec}, classes={d_output}")

    net = GTNParallel3Domains(
        d_temp=d_temp,
        d_wave=d_wave,
        d_spec=d_spec,
        d_input_L=d_input,
        d_model=d_model,
        d_hidden=d_hidden,
        q=q, v=v, h=h, N=N,
        dropout=dropout,
        device=DEVICE,
        d_output=d_output,
        pe=pe,
        mask=mask
    ).to(DEVICE)

    with torch.no_grad():
        xb_time, xb_wave, xb_spec, yb = next(iter(train_dataloader))
        logits, *_ = net(
            xb_time.to(DEVICE),
            xb_wave.to(DEVICE),
            xb_spec.to(DEVICE),
            'test'
        )
        print("OK forward:", tuple(logits.shape))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=LR)

    train(net, train_dataloader, val_dataloader, test_dataloader,
          DEVICE, EPOCH, test_interval, BATCH_SIZE,
          reslut_figure_path, save_path, file_name,
          criterion, optimizer,
          d_model, q, v, h, N, dropout, DATA_LEN, draw_key,
          optimizer_name, LR, pe, mask)
