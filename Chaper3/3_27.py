# 3_27.py
import numpy as np

w = np.array([1, 2, -10])
x = np.array([3, 4, 1])
y_that = -1

# 1. Tinh w^T x
diem = np.dot(w, x)
print("w^T x =", diem)

# 2. Nhan du doan
if diem >= 0:
    nhan_du_doan = 1
else:
    nhan_du_doan = -1
print("Nhan du doan =", nhan_du_doan)

# 3. Kiem tra phan lop sai
if y_that * diem <= 0:
    print("Diem du lieu bi phan lop SAI")
else:
    print("Diem du lieu duoc phan lop DUNG")