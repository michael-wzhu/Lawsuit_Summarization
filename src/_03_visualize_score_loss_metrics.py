import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df_loss = pd.read_csv('../output/infos/loss.csv', header=0)
df_metrics = pd.read_csv("../output/infos/metrics.csv", header=0)

loss_train = df_loss["train_loss"].values
loss_valid = df_loss["valid_loss"].values

train_p = df_metrics["train_p"].values
train_r = df_metrics["train_r"].values
train_f1 = df_metrics["train_f1"].values
train_acc = df_metrics["train_acc"].values

valid_p = df_metrics["valid_p"].values
valid_r = df_metrics["valid_r"].values
valid_f1 = df_metrics["valid_f1"].values
valid_acc = df_metrics["valid_acc"].values

val_zi = df_metrics["val_zi"].values
# val_r1_zi = df_metrics["val_r1_zi"].values
# val_r2_zi = df_metrics["val_r2_zi"].values
# val_rl_zi = df_metrics["val_rl_zi"].values
val_ci = df_metrics["val_ci"].values
# val_r1_ci = df_metrics["val_r1_ci"].values
# val_r2_ci = df_metrics["val_r2_ci"].values
# val_rl_ci = df_metrics["val_rl_ci"].values


num_epoch = len(loss_train)
steps = np.arange(1, num_epoch +1)

plt.subplot(2, 2, 1)
plt.plot(steps, loss_train, 'b--', label="train loss")
plt.plot(steps, loss_valid, 'r--', label="valid loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(steps, val_zi, 'kx-.', label="rouge based on single words")
plt.plot(steps, val_ci, 'ko:', label="rouge based on words")
plt.xlabel('epoch')
plt.ylabel('rouge score')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(steps, train_p, 'g--', label="train precision")
plt.plot(steps, train_r, 'y--', label="train recall")
plt.plot(steps, train_f1, 'bo--', label="train f1-score")
plt.plot(steps, train_acc, 'r--', label="train accuracy")
plt.xlabel('epoch')
plt.ylabel('train metrics')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(steps, valid_p, 'g--', label="valid precision")
plt.plot(steps, valid_r, 'y--', label="valid recall")
plt.plot(steps, valid_f1, 'bo--', label="valid f1-score")
plt.plot(steps, valid_acc, 'r--', label="valid accuracy")
plt.xlabel('epoch')
plt.ylabel('valid metrics')
plt.legend()

plt.show()
