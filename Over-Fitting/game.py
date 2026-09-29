import numpy as np
import pandas as pd
 
CSV_FILE = "data.csv"
FEATURE_COLS = ["Critic_Score"]
TARGET_COL = "Global_Sales"
N_TRAIN = 5          # CỐ TÌNH để rất ít mẫu train -> minh hoạ overfitting
RANDOM_SEED = 42
 
# 1. Đọc dữ liệu
df = pd.read_csv(CSV_FILE)
X = df[FEATURE_COLS].values.astype(float)   # (N, d)
y = df[TARGET_COL].values.astype(float)     # (N,)
 
# 2. Chia train/test — train CHỈ N_TRAIN mẫu, còn lại làm test
rng = np.random.default_rng(RANDOM_SEED)
N = len(y)
indices = rng.permutation(N)
train_idx, test_idx = indices[:N_TRAIN], indices[N_TRAIN:]
 
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
 
# 3. Thêm hàng bias (số 1) -> X dạng (d+1, N) theo đúng công thức
ones_train = np.ones((1, len(y_train)))
Xt_train = np.vstack([ones_train, X_train.T])           # (d+1, N_train)
 
# 4. Công thức: w = (X Xᵀ)† X y  (chỉ dùng công thức thường, không có regularization)
w = np.linalg.pinv(Xt_train @ Xt_train.T) @ Xt_train @ y_train
print("w =", w)
 
# 5. Train error (trên tập train) — chỉ dùng bình phương sai số (MSE)
y_train_pred = w @ Xt_train
train_mse = np.mean((y_train - y_train_pred) ** 2)
 
# 6. Test error (trên tập test — dữ liệu mô hình CHƯA từng thấy)
ones_test = np.ones((1, len(y_test)))
Xt_test = np.vstack([ones_test, X_test.T])               # (d+1, N_test)
y_test_pred = w @ Xt_test
test_mse = np.mean((y_test - y_test_pred) ** 2)
 
print(f"Train MSE = {train_mse:.4f}")
print(f"Test MSE  = {test_mse:.4f}")