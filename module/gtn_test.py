# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:41:51 2025

@author: michel
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from dataset_process.dataset_process import MyDataset
from module.transformer import Transformer
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization1 import result_visualization #, plot_training_results
from time import time
from tqdm import tqdm

# Set random seed and environment
setup_seed(30)
reslut_figure_path = 'result_figure'

# Dataset
path = r'E:\Michel\GTN-master\Gated Transformer 论文IJCAI版\GTN_dataset_structured.mat'
test_interval = 5
draw_key = 1
file_name2 = path.split('\\')[-1][0:path.split('\\')[-1].index('.')]

# Hyperparameters
EPOCH = 80
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

d_model = 256
d_hidden = 128
q = 8
v = 8
h = 8
N = 8
dropout = 0.2
pe = True
mask = True
optimizer_name = 'Adam'

# Load data
train_dataset = MyDataset(path, 'train')
test_dataset = MyDataset(path, 'test')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

DATA_LEN = train_dataset.train_len
d_input = train_dataset.input_len
d_channel = train_dataset.channel_len
d_output = train_dataset.output_len

# Model
net = Transformer(d_model, d_input, d_channel, d_output, d_hidden, q, v, h, N, DEVICE, dropout, pe, mask).to(DEVICE)
loss_function = Myloss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

# Tracking
correct_on_train, correct_on_test, loss_list = [], [], []
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

# Train function
def train():
    global time_cost
    net.train()
    max_accuracy = 0
    begin = time()
    for epoch in range(EPOCH):
        net.train()
        epoch_loss = 0.0
        batch_count = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCH}", unit="batch") as tepoch:
            for x, y in tepoch:
                optimizer.zero_grad()
                y_pre, *_ = net(x.to(DEVICE), 'train')
                loss = loss_function(y_pre, y.to(DEVICE))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
                tepoch.set_postfix(loss=loss.item())
        mean_loss = epoch_loss / batch_count
        loss_list.append(mean_loss)
        print(f"Epoch {epoch+1}: Mean Loss = {mean_loss:.4f}")
        if (epoch + 1) % test_interval == 0:
            current_accuracy = test(test_dataloader)
            test(train_dataloader, 'train_set')
            print(f"Max Accuracy so far — Test: {max(correct_on_test)}% | Train: {max(correct_on_train)}%")
            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                torch.save(net, f'E:/Michel/GTN-master/Gated Transformer 论文IJCAI版/saved_model/{file_name2} batch={BATCH_SIZE}.pkl')
    try:
        os.rename(f'E:/Michel/GTN-master/Gated Transformer 论文IJCAI版/saved_model/{file_name2} batch={BATCH_SIZE}.pkl',
                  f'E:/Michel/GTN-master/Gated Transformer 论文IJCAI版/saved_model/{file_name2} {max_accuracy:.2f}% batch={BATCH_SIZE}.pkl')
    except Exception as e:
        print(f"[⚠️ os.rename failed] {e}")
    end = time()
    time_cost = round((end - begin) / 60, 2)
    print(f"⏱️ Training completed in {time_cost} min.")
    result_visualization(
        loss_list=loss_list,
        correct_on_test=correct_on_test,
        correct_on_train=correct_on_train,
        test_interval=test_interval,
        d_model=d_model, q=q, v=v, h=h, N=N,
        dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
        time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key,
        reslut_figure_path=reslut_figure_path, file_name=file_name2,
        optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask
    )
    evaluate_and_plot_confusion_matrix(net, test_dataloader, DEVICE,
        ['background', 'digging', 'knocking', 'watering', 'shaking', 'walking'],
        os.path.join(reslut_figure_path, 'GTN_confusion_matrix.jpg'))

# Evaluation with confusion matrix
def evaluate_and_plot_confusion_matrix(model, dataloader, device, class_labels, save_path):
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
    sns.heatmap(df, fmt='g', annot=True, robust=True,
                annot_kws={'size': 10},
                xticklabels=class_labels,
                yticklabels=class_labels,
                cmap='Reds')
    plt.xlabel('Predicted label', fontsize=15)
    plt.ylabel('True label', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    # Metrics
    Acc = np.trace(C) / np.sum(C)
    NAR = (np.sum(C[0]) - C[0][0]) / np.sum(C[:, 1:]) if np.sum(C[:, 1:]) != 0 else 0
    FNR = (np.sum(C[:, 0]) - C[0][0]) / np.sum(C[1:, :]) if np.sum(C[1:, :]) != 0 else 0

    column_sum = np.sum(C, axis=0)
    row_sum = np.sum(C, axis=1)

    for i in range(len(class_labels)):
        TP = C[i, i]
        Precision = TP / column_sum[i] if column_sum[i] != 0 else 0
        Recall = TP / row_sum[i] if row_sum[i] != 0 else 0
        F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) != 0 else 0
        print(f'Precision_{i}: {Precision:.3f}')
        print(f'Recall_{i}: {Recall:.3f}')
        print(f'F1_{i}: {F1:.3f}')

    print(f'Accuracy: {Acc:.4f}')
    print(f'NAR: {NAR:.4f}')
    print(f'FNR: {FNR:.4f}')

if __name__ == '__main__':
    train()
