import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from Perceptron import Perceptron

# ---------- 1. Đọc và chia dữ liệu ----------
data = load_breast_cancer()
X = data.data
y = data.target
# Đổi nhãn 0 -> -1 để dùng với Perceptron
for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ---------- 2. Chuẩn hóa (chỉ tính trên tập train) ----------
trung_binh = X_train.mean(axis=0)
do_lech = X_train.std(axis=0)
X_train = (X_train - trung_binh) / do_lech
X_val = (X_val - trung_binh) / do_lech
X_test = (X_test - trung_binh) / do_lech

# ---------- 3. Các độ đo (lớp dương là +1) ----------
def tinh_do_do(y_that, y_du_doan):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_that)):
        if y_that[i] == 1 and y_du_doan[i] == 1:
            tp = tp + 1
        elif y_that[i] == -1 and y_du_doan[i] == -1:
            tn = tn + 1
        elif y_that[i] == -1 and y_du_doan[i] == 1:
            fp = fp + 1
        else:
            fn = fn + 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return accuracy, precision, recall, f1

# ---------- 4. Tìm tham số tốt nhất trên tập validation ----------
danh_sach_lr = [0.001, 0.01, 0.1, 1]
danh_sach_epoch = [10, 50, 100, 500]

f1_tot_nhat = -1
lr_tot_nhat = None
epoch_tot_nhat = None

for lr in danh_sach_lr:
    for so_epoch in danh_sach_epoch:
        model = Perceptron(learning_rate=lr, max_epochs=so_epoch)
        model.fit(X_train, y_train)
        y_val_du_doan = model.predict(X_val)
        acc, pre, rec, f1 = tinh_do_do(y_val, y_val_du_doan)
        print("lr =", lr, "epoch =", so_epoch, "F1 =", round(f1, 4))
        if f1 > f1_tot_nhat:
            f1_tot_nhat = f1
            lr_tot_nhat = lr
            epoch_tot_nhat = so_epoch

print("Tham số tốt nhất: lr =", lr_tot_nhat, ", epoch =", epoch_tot_nhat)

# ---------- 5. Huấn luyện lại và đánh giá trên tập test ----------
X_train_val = np.vstack((X_train, X_val))
y_train_val = np.concatenate((y_train, y_val))

model = Perceptron(learning_rate=lr_tot_nhat, max_epochs=epoch_tot_nhat)
model.fit(X_train_val, y_train_val)
y_test_du_doan = model.predict(X_test)

acc, pre, rec, f1 = tinh_do_do(y_test, y_test_du_doan)
print("Accuracy :", round(acc, 4))
print("Precision:", round(pre, 4))
print("Recall   :", round(rec, 4))
print("F1-score :", round(f1, 4))