# -*- coding: utf-8 -*-
"""
Created on Tue May  6 11:06:36 2025

@author: michel
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

#coding = UTF-8


from scipy.io import loadmat



import datetime
from sklearn import svm, preprocessing
# from get_das_data import get_das_data,get_stats_features
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import seaborn as sns

# from transformer import ViT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




from torch.utils.data import DataLoader
from dataset_process.dataset_process import MyDataset
import torch.optim as optim
from time import time
from tqdm import tqdm
import os
from transformer import Transformer
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization

# === ADDED: plot train/val accuracy & loss au style de l’exemple ===
def draw_loss_acc(train_acc, train_loss, val_acc, val_loss, out_png: str) -> None:
    """Plot training/validation accuracy and loss over epochs (style identique à l'exemple)."""
    import os, numpy as np
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # Sécurité : tronquer toutes les séries à la même longueur
    n = min(len(train_acc), len(val_acc), len(train_loss), len(val_loss))
    train_acc, val_acc   = list(train_acc[:n]), list(val_acc[:n])
    train_loss, val_loss = list(train_loss[:n]), list(val_loss[:n])

    epochs = range(n)

    plt.figure(figsize=(8, 8))

    # ---- ACCURACY ----
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_acc, label="train", linewidth=1.5)
    plt.plot(epochs, val_acc,   label="val",   linewidth=1.5)
    plt.title('Accuracy vs. epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

    # ---- LOSS ----
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_loss, label="train", linewidth=1.5)
    plt.plot(epochs, val_loss,   label="val",   linewidth=1.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    ymin = 0.0
    ymax = max(1.0, max(train_loss + val_loss)) * 1.05
    plt.ylim(ymin, ymax)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

    # Repères réguliers sur l’axe des epochs
    try:
        ticks = np.linspace(0, n-1, 6, dtype=int) if n >= 5 else range(n)
        plt.subplot(2, 1, 1); plt.xticks(ticks)
        plt.subplot(2, 1, 2); plt.xticks(ticks)
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()
    plt.close()

# from mytest.gather.main import draw

setup_seed(30)  # Set random seed
reslut_figure_path = r'D:\Michel\DAStatFormer\results_figure'  # Path to save result figures


path = r'D:\Michel\Gated Transformer 论文IJCAI版\DAS_DataNorm_24_features_selected.mat' #r'D:\Michel\DAStatFormer\DAS_DataNorm_all_domains_for_GTN.mat' #r'E:\Michel\GTN-master\Gated Transformer 论文IJCAI版\DAS_Data_for_GTN_12_24.mat' 
# path = r"E:\Michel\GTN-master\Gated Transformer 论文IJCAI版\DAS_Data_Time_for_GTN_24_11.mat"
test_interval = 5  # Test interval in epochs
draw_key = 1  # Save visualizations only if epoch >= draw_key
file_name = path.split('\\')[-1][0:path.split('\\')[-1].index('.')]  # Extract file name

# Hyperparameter settings
EPOCH = 100
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Select device: CPU or GPU
print(f'Using device: {DEVICE}')

d_model = 256 # 512
d_hidden = 128
q = 8
v = 8
h = 8
N = 8
dropout = 0.2
pe = True  # Use positional encoding in one of the towers (score=pe)
mask = True  # Use input masking in one of the towers (score=input)
optimizer_name = 'Adagrad'  # Choose optimizer



# # Load dataset
# train_dataset = MyDataset(path, 'train')
# test_dataset = MyDataset(path, 'test')
# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# # st_dataloader = DataLoader(dataset=Xtest_gtn, batch_size=BATCH_SIZE, shuffle=False)

# # Extract dataset dimensions
# DATA_LEN = train_dataset.train_len        # Number of training samples
# d_input = train_dataset.input_len         # Number of time steps
# d_channel = train_dataset.channel_len     # Number of input features (channels)
# d_output = train_dataset.output_len       # Number of output classes

from torch.utils.data import random_split, DataLoader


base_train_ds = MyDataset(path, 'train')
test_dataset  = MyDataset(path, 'test')


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


# # Build the Transformer model


net = Transformer(
    d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
    q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE
).to(DEVICE)


# Define the loss function (cross-entropy)
loss_function = Myloss()


# Choose optimizer
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# Lists to track metrics
correct_on_train = []
correct_on_test = []
loss_list = []
time_cost = 0

# Test function
def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, *_ = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
        accuracy = round((100 * correct / total), 2)
        if flag == 'test_set':
            correct_on_test.append(accuracy)
        elif flag == 'train_set':
            correct_on_train.append(accuracy)
        print(f'Accuracy on {flag}: {accuracy:.2f} %')
        return accuracy

# Training function

save_path = r"D:\Michel\DAStatFormer\results_figure"



def evaluate_and_plot_confusion_matrix(model, dataloader, device, class_labels):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pre, *_ = model(x, 'test')
            preds = y_pre.argmax(dim=1).cpu().numpy()
            labels = y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    C = confusion_matrix(all_labels, all_preds)
    df = pd.DataFrame(C)

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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(f'{save_path}/{file_name}_Conf_Max.png')  #(save_path+"/DAStatFormer_Time_domain_confusion_matrix.jpg") 
    plt.show()
    plt.close()
    
    # === MÉTRIQUES CALCULÉES ===
    print("\n===== Performance Metrics =====")
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
        recall = TP / row_sum[i] if row_sum[i] != 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
        print(f'Precision_{i}: {precision:.3f}')
        print(f'Recall_{i}:    {recall:.3f}')
        print(f'F1_{i}:        {f1:.3f}')



train_dataloader


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔍 Total parameters       : {total_params:,}")
    print(f"🧠 Trainable parameters   : {trainable_params:,}")
    print(f"💾 Estimated model size   : {trainable_params * 4 / 1024 ** 2:.2f} MB (float32)\n")

def train():
    print("\n📊 ==== Model Summary ====")
    count_parameters(net)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # Reset all peak memory stats

    net.train()
    max_accuracy = 0
    begin = time()
    
    # === ADDED: buffers pour courbes epoch-par-epoch ===
    train_losses_plot, val_losses_plot = [], []
    train_acc_plot,    val_acc_plot    = [], []

    for epoch in range(EPOCH):
        net.train()
        epoch_loss = 0.0
        batch_count = 0

        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCH}", unit="batch") as tepoch:
            for x, y in tepoch:
                optimizer.zero_grad()
                y_pre, *_ = net(x.to(DEVICE), 'train')
                loss = criterion(y_pre, y.to(DEVICE))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                # Mémoire RAM et GPU en temps réel
                import psutil
                ram_usage = psutil.virtual_memory().used / 1024 ** 3  # GB
                postfix = {"loss": f"{loss.item():.4f}", "RAM": f"{ram_usage:.2f}GB"}

                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated(DEVICE) / 1024 ** 2
                    reserv = torch.cuda.memory_reserved(DEVICE) / 1024 ** 2
                    postfix.update({"GPU_alloc": f"{alloc:.1f}MB", "GPU_resv": f"{reserv:.1f}MB"})

                tepoch.set_postfix(postfix)

        mean_loss = epoch_loss / batch_count
        loss_list.append(mean_loss)
        # print(f"Epoch {epoch+1}: Mean Loss = {mean_loss:.4f}")
        
        # print(f"Epoch {epoch+1}: Mean Loss = {mean_loss:.4f}")
        
        # # === ADDED: évaluation légère pour alimenter les 4 courbes à chaque epoch ===
        # tr_loss_ep, tr_acc_ep = evaluate_loss_acc(net, train_dataloader, DEVICE, desc="train")
        # va_loss_ep, va_acc_ep = evaluate_loss_acc(net, test_dataloader,  DEVICE, desc="valid")
        
        # train_losses_plot.append(tr_loss_ep)
        # val_losses_plot.append(va_loss_ep)
        # train_acc_plot.append(tr_acc_ep)
        # val_acc_plot.append(va_acc_ep)
        
        # # === ADDED: figure epochique (optionnel ; commente si tu veux seulement la finale) ===
        # try:
        #     out_png_epoch = os.path.join(reslut_figure_path, f"{file_name}_loss_acc_epoch_{epoch+1:03d}.png")
        #     draw_loss_acc(train_acc_plot, train_losses_plot, val_acc_plot, val_losses_plot, out_png_epoch)
        # except Exception as e:
        #     print(f"[plot at epoch {epoch+1} failed] {e}")
        
        print(f"Epoch {epoch+1}: Mean Loss = {mean_loss:.4f}")

        # ...
        tr_loss_ep, tr_acc_ep = evaluate_loss_acc(net, train_dataloader, DEVICE, desc="train")
        va_loss_ep, va_acc_ep = evaluate_loss_acc(net, val_dataloader, DEVICE, desc="valid")
        # ...
        if (epoch + 1) % test_interval == 0:
            current_accuracy = va_acc_ep   # validation accuracy, pas test
            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                torch.save(net.state_dict(), f'D:/Michel/DAStatFormer/saved_model/{file_name}_best.pkl')

        # tr_loss_ep, tr_acc_ep = evaluate_loss_acc(net, train_dataloader, DEVICE, desc="train")
        # va_loss_ep, va_acc_ep = evaluate_loss_acc(net, test_dataloader,  DEVICE, desc="valid")
        
        # train_losses_plot.append(tr_loss_ep)
        # val_losses_plot.append(va_loss_ep)
        # train_acc_plot.append(tr_acc_ep)
        # val_acc_plot.append(va_acc_ep)



        # # Test tous les X epochs
        # if (epoch + 1) % test_interval == 0:
        #     current_accuracy = test(test_dataloader)
        #     test(train_dataloader, 'train_set')
        #     print(f"Max Accuracy so far — Test: {max(correct_on_test)}% | Train: {max(correct_on_train)}%")

        #     if current_accuracy > max_accuracy:
        #         max_accuracy = current_accuracy
        #         torch.save(net, f'D:/Michel/DAStatFormer/saved_model/{file_name} batch={BATCH_SIZE}.pkl')

    # Optionnel : renommer le meilleur modèle
    try:
        os.rename(f'D:/Michel/DAStatFormer/saved_model/{file_name} batch={BATCH_SIZE}.pkl',
                  f'D:/Michel/DAStatFormer/saved_model/{file_name} {max_accuracy:.2f}% batch={BATCH_SIZE}.pkl')
    except Exception as e:
        print(f"[⚠️ os.rename failed] {e}")

    end = time()
    time_cost = round((end - begin) / 60, 2)
    print(f"\n⏱️ Training completed in {time_cost} min.")

    # Mémoire GPU finale
    if torch.cuda.is_available():
        max_alloc = torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2
        max_reserv = torch.cuda.max_memory_reserved(DEVICE) / 1024 ** 2
        print(f"📈 Max GPU Memory Allocated: {max_alloc:.2f} MB")
        print(f"📦 Max GPU Memory Reserved : {max_reserv:.2f} MB")

    # === ADDED: figure finale loss/acc ===
    try:
        final_png = os.path.join(reslut_figure_path, f"{file_name}_loss_acc.png")
        draw_loss_acc(train_acc_plot, train_losses_plot, val_acc_plot, val_losses_plot, final_png)
        print(f"[OK] Courbes loss/acc sauvegardées : {final_png}")
    except Exception as e:
        print(f"[final plot failed] {e}")


    evaluate_and_plot_confusion_matrix(
        model=net,
        dataloader=test_dataloader,
        device=DEVICE,
        class_labels = ['0','1','2','3','4','5']  #['background', 'digging', 'knocking', 'watering', 'shaking', 'walking'],
    )

        # === Measure inference speed ===
    avg_infer_time_ms = measure_inference_time(net, test_dataloader, DEVICE)
    print(f"Average inference time per sample: {avg_infer_time_ms:.3f} ms")

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

from pathlib import Path
import matplotlib.pyplot as plt

def plot_curve(title, xlabel, ylabel, pic_file, curve1, curve2=None, legend=None):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot(curve1)
    if curve2 is not None:
        plt.plot(curve2)
        if legend:
            plt.legend(legend, loc='best')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    Path(pic_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pic_file, dpi=160)
    plt.close()

def evaluate_loss_acc(model, dataloader, device, desc="valid"):
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

            # Warmup (on ignore les premières itérations)
            if i < n_warmup:
                _ = model(x, 'test')
                continue

            start_time = tm.perf_counter()
            _ = model(x, 'test')
            end_time = tm.perf_counter()

            elapsed = end_time - start_time
            times.append(elapsed)
            total_samples += batch_size

    # Moyenne par batch et par échantillon
    total_time = np.sum(times)
    avg_time_per_batch = total_time / len(times)
    avg_time_per_sample = (total_time / total_samples) * 1000  # en ms

    print(f"\n⚡ Inference time per batch:   {avg_time_per_batch:.4f} s")
    print(f"⚡ Inference time per sample: {avg_time_per_sample:.4f} ms\n")

    return avg_time_per_sample


if __name__ == '__main__':
    # train_dynamic()
    train()

