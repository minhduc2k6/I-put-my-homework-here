# 3_28.py
import numpy as np

w = np.array([-2.0, 1.0, 0.0])
x = np.array([2.0, 3.0, 1.0])
y = 1
learning_rate = 1

# 1. Kiem tra mau co bi phan lop sai khong
diem_truoc = np.dot(w, x)
print("w^T x truoc cap nhat =", diem_truoc)

if y * diem_truoc <= 0:
    print("Mau bi phan lop SAI")

    # 2. Cap nhat Perceptron: w_moi = w + y * x
    w = w + learning_rate * y * x
    print("w sau cap nhat =", w)

    # 3. Tinh lai w^T x
    diem_sau = np.dot(w, x)
    print("w^T x sau cap nhat =", diem_sau)
else:
    print("Mau duoc phan lop DUNG, khong can cap nhat")