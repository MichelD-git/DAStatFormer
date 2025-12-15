# # -*- coding: utf-8 -*-
# """
# Created on Mon Jul  7 11:00:33 2025

# @author: michel
# """

# # import matplotlib.pyplot as plt
# # from matplotlib.font_manager import FontProperties as fp  # 1、引入FontProperties
# # import math

# # def result_visualization(loss_list: list,
# #                          correct_on_test: list,
# #                          correct_on_train: list,
# #                          test_interval: int,
# #                          d_model: int,
# #                          q: int,
# #                          v: int,
# #                          h: int,
# #                          N: int,
# #                          dropout: float,
# #                          DATA_LEN: int,
# #                          BATCH_SIZE: int,
# #                          time_cost: float,
# #                          EPOCH: int,
# #                          draw_key: int,
# #                          reslut_figure_path: str,
# #                          optimizer_name: str,
# #                          file_name: str,
# #                          LR: float,
# #                          pe: bool,
# #                          mask: bool):
# #     my_font = fp(fname=r"font/simsun.ttc")  # 2、设置字体路径

# #     # 设置风格
# #     plt.style.use('seaborn-v0_8')

# #     fig = plt.figure()  # 创建基础图
# #     ax1 = fig.add_subplot(311)  # 创建两个子图
# #     ax2 = fig.add_subplot(313)

# #     ax1.plot(loss_list)  # 添加折线
# #     ax2.plot(correct_on_test, color='red', label='on Test Dataset')
# #     ax2.plot(correct_on_train, color='blue', label='on Train Dataset')

# #     # 设置坐标轴标签 和 图的标题
# #     ax1.set_xlabel('epoch')
# #     ax1.set_ylabel('loss')
# #     ax2.set_xlabel(f'epoch/{test_interval}')
# #     ax2.set_ylabel('correct')
# #     ax1.set_title('LOSS')
# #     ax2.set_title('CORRECT')

# #     plt.legend(loc='best')

# #     # 设置文本
# #     fig.text(x=0.13, y=0.4, s=f'最小loss：{min(loss_list)}' '    '
# #                               f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}' '    '
# #                               f'最后一轮loss:{loss_list[-1]}' '\n'
# #                               f'最大correct：测试集:{max(correct_on_test)}% 训练集:{max(correct_on_train)}%' '    '
# #                               f'最大correct对应的已训练epoch数:{(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}' '    '
# #                               f'最后一轮correct：{correct_on_test[-1]}%' '\n'
# #                               f'd_model={d_model}   q={q}   v={v}   h={h}   N={N}  drop_out={dropout}'  '\n'
# #                               f'共耗时{round(time_cost, 2)}分钟', fontproperties=my_font)

# #     # 保存结果图   测试不保存图（epoch少于draw_key）
# #     if EPOCH >= draw_key:
# #         plt.savefig(
# #             f'{reslut_figure_path}/{file_name} {max(correct_on_test)}% {optimizer_name} epoch={EPOCH} batch={BATCH_SIZE} lr={LR} pe={pe} mask={mask} [{d_model},{q},{v},{h},{N},{dropout}].png')

# #     # 展示图
# #     plt.show()

# #     print('正确率列表', correct_on_test)

# #     print(f'最小loss：{min(loss_list)}\r\n'
# #           f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}\r\n'
# #           f'最后一轮loss:{loss_list[-1]}\r\n')

# #     print(f'最大correct：测试集:{max(correct_on_test)}\t 训练集:{max(correct_on_train)}\r\n'
# #           f'最correct对应的已训练epoch数:{(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}\r\n'
# #           f'最后一轮correct:{correct_on_test[-1]}')

# #     print(f'共耗时{round(time_cost, 2)}分钟')

# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties as fp  # 1. Import FontProperties
# import math

# def result_visualization(loss_list: list,
#                          correct_on_test: list,
#                          correct_on_train: list,
#                          test_interval: int,
#                          d_model: int,
#                          q: int,
#                          v: int,
#                          h: int,
#                          N: int,
#                          dropout: float,
#                          DATA_LEN: int,
#                          BATCH_SIZE: int,
#                          time_cost: float,
#                          EPOCH: int,
#                          draw_key: int,
#                          reslut_figure_path: str,
#                          optimizer_name: str,
#                          file_name: str,
#                          LR: float,
#                          pe: bool,
#                          mask: bool):
#     try:
#         my_font = fp(fname=r"font/simsun.ttc")  
#     except Exception:
#         print("⚠️ Police 'simsun.ttc' non trouvée, police par défaut utilisée.")
#         my_font = None
#     # Set style
#     plt.style.use('seaborn-v0_8')

#     fig = plt.figure()  # Create base figure
#     ax1 = fig.add_subplot(311)  # Create two subplots
#     ax2 = fig.add_subplot(313)

#     ax1.plot(loss_list)  # Add loss curve
#     ax2.plot(correct_on_test, color='red', label='on Test Dataset')
#     ax2.plot(correct_on_train, color='blue', label='on Train Dataset')

#     # Set axis labels and titles
#     ax1.set_xlabel('epoch')
#     ax1.set_ylabel('loss')
#     ax2.set_xlabel(f'epoch/{test_interval}')
#     ax2.set_ylabel('correct')
#     ax1.set_title('LOSS')
#     ax2.set_title('CORRECT')

#     plt.legend(loc='best')

#     # Set text description
#     fig.text(x=0.13, y=0.4, s=f'Min loss: {min(loss_list)}' '    '
#                               f'Epoch of min loss: {math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}' '    '
#                               f'Final loss: {loss_list[-1]}' '\n'
#                               f'Max accuracy: Test {max(correct_on_test)}% | Train {max(correct_on_train)}%' '    '
#                               f'Epoch of max test accuracy: {(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}' '    '
#                               f'Final test accuracy: {correct_on_test[-1]}%' '\n'
#                               f'd_model={d_model}   q={q}   v={v}   h={h}   N={N}  drop_out={dropout}'  '\n'
#                               f'Total training time: {round(time_cost, 2)} min', fontproperties=my_font)

#     # Save result figure (only if epoch >= draw_key)
#     if EPOCH >= draw_key:
#         plt.savefig(
#             f'{reslut_figure_path}/{file_name} {max(correct_on_test)}% {optimizer_name} epoch={EPOCH} batch={BATCH_SIZE} lr={LR} pe={pe} mask={mask} [{d_model},{q},{v},{h},{N},{dropout}].png')

#     # Show plot
#     plt.show()

#     print('Test accuracy list:', correct_on_test)

#     print(f'Min loss: {min(loss_list)}\r\n'
#           f'Epoch of min loss: {math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}\r\n'
#           f'Final loss: {loss_list[-1]}\r\n')

#     print(f'Max accuracy: Test {max(correct_on_test)}\t Train {max(correct_on_train)}\r\n'
#           f'Epoch of max test accuracy: {(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}\r\n'
#           f'Final test accuracy: {correct_on_test[-1]}')

#     print(f'Total training time: {round(time_cost, 2)} minutes')

import os
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as fp
import numpy as np

def result_visualization(loss_list: list,
                         correct_on_test: list,
                         correct_on_train: list,
                         test_interval: int,
                         d_model: int,
                         q: int,
                         v: int,
                         h: int,
                         N: int,
                         dropout: float,
                         DATA_LEN: int,
                         BATCH_SIZE: int,
                         time_cost: float,
                         EPOCH: int,
                         draw_key: int,
                         reslut_figure_path: str,
                         optimizer_name: str,
                         file_name: str,
                         LR: float,
                         pe: bool,
                         mask: bool,
                         confusion_matrix_np: np.ndarray = None,
                         class_labels: list = None):
    
    try:
        my_font = fp(fname=r"font/simsun.ttc")
    except Exception:
        print("⚠️ Police 'simsun.ttc' non trouvée, police par défaut utilisée.")
        my_font = None

    # Set style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(313)

    ax1.plot(loss_list)
    ax2.plot(correct_on_test, color='red', label='on Test Dataset')
    ax2.plot(correct_on_train, color='blue', label='on Train Dataset')

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel(f'epoch/{test_interval}')
    ax2.set_ylabel('accuracy (%)')
    ax1.set_title('LOSS')
    ax2.set_title('ACCURACY')
    plt.legend(loc='best')

    # Texte résumé
    fig.text(x=0.13, y=0.4, s=f'Min loss: {min(loss_list)}    '
                              f'Epoch of min loss: {math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}    '
                              f'Final loss: {loss_list[-1]}\n'
                              f'Max accuracy: Test {max(correct_on_test)}% | Train {max(correct_on_train)}%    '
                              f'Epoch of max test accuracy: {(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}    '
                              f'Final test accuracy: {correct_on_test[-1]}%\n'
                              f'd_model={d_model}   q={q}   v={v}   h={h}   N={N}  drop_out={dropout}\n'
                              f'Total training time: {round(time_cost, 2)} min',
             fontproperties=my_font)

    # Enregistrement de la figure
    if EPOCH >= draw_key:
        os.makedirs(reslut_figure_path, exist_ok=True)
        fig_path = os.path.join(reslut_figure_path, f'{file_name} {max(correct_on_test)}% {optimizer_name} epoch={EPOCH} batch={BATCH_SIZE} lr={LR} pe={pe} mask={mask} [{d_model},{q},{v},{h},{N},{dropout}].png')
        plt.savefig(fig_path, dpi=300)

    plt.show()

    # Impression terminal
    print('Test accuracy list:', correct_on_test)
    print(f'Min loss: {min(loss_list)}')
    print(f'Epoch of min loss: {math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}')
    print(f'Final loss: {loss_list[-1]}')
    print(f'Max accuracy: Test {max(correct_on_test)}\t Train {max(correct_on_train)}')
    print(f'Epoch of max test accuracy: {(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}')
    print(f'Final test accuracy: {correct_on_test[-1]}')
    print(f'Total training time: {round(time_cost, 2)} minutes')

    # ========= Enregistrement dans un fichier texte ============
    try:
        os.makedirs(reslut_figure_path, exist_ok=True)
        summary_path = os.path.join(reslut_figure_path, f"{file_name}_metrics_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("===== Performance Metrics =====\n")
            f.write(f"Accuracy: {correct_on_test[-1] / 100:.4f}\n")
            f.write(f"Min loss: {min(loss_list)}\n")
            f.write(f"Epoch of min loss: {math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}\n")
            f.write(f"Final loss: {loss_list[-1]}\n\n")

            f.write(f"Max accuracy: Test {max(correct_on_test)}\t Train {max(correct_on_train)}\n")
            f.write(f"Epoch of max test accuracy: {(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}\n")
            f.write(f"Final test accuracy: {correct_on_test[-1]}\n")
            f.write(f"Total training time: {round(time_cost, 2)} minutes\n\n")

            # Si la matrice de confusion est fournie :
            if confusion_matrix_np is not None:
                C = confusion_matrix_np
                f.write("===== Confusion Matrix Stats =====\n")
                Acc = np.trace(C) / np.sum(C)
                NAR = (np.sum(C[0]) - C[0][0]) / np.sum(C[:, 1:])
                FNR = (np.sum(C[:, 0]) - C[0][0]) / np.sum(C[1:, :])

                f.write(f"Accuracy: {Acc:.4f}\n")
                f.write(f"NAR: {NAR:.4f}\n")
                f.write(f"FNR: {FNR:.4f}\n")
                column_sum = np.sum(C, axis=0)
                row_sum = np.sum(C, axis=1)
                f.write(f"Column sums: {column_sum.tolist()}\n")
                f.write(f"Row sums: {row_sum.tolist()}\n\n")

                for i in range(len(C)):
                    P = C[i][i] / column_sum[i] if column_sum[i] != 0 else 0
                    R = C[i][i] / row_sum[i] if row_sum[i] != 0 else 0
                    F1 = 2 * P * R / (P + R) if P + R != 0 else 0
                    label = class_labels[i] if class_labels else f"Class_{i}"
                    f.write(f"Precision_{label}: {P:.3f}\n")
                    f.write(f"Recall_{label}:    {R:.3f}\n")
                    f.write(f"F1_{label}:        {F1:.3f}\n\n")
        print(f"📄 Résumé enregistré dans : {summary_path}")
    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde du résumé texte : {e}")
