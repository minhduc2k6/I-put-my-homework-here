import numpy as np
from perceptron_model import Perceptron

# Tạo dữ liệu 2 lớp giống bài mẫu, nhưng mỗi điểm là một HÀNG
np.random.seed(2)
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)   # lớp +1
X1 = np.random.multivariate_normal(means[1], cov, N)   # lớp -1

X = np.concatenate((X0, X1), axis=0)                   # (20, 2)
y = np.concatenate((np.ones(N), -np.ones(N)))          # (20,)

model = Perceptron(max_iter=1000)
model.fit(X, y)

print("w học được (bias, w1, w2):", model.w)
print("Số lần cập nhật:", model.n_updates)

y_pred = model.predict(X)
print("Độ chính xác trên dữ liệu train:", np.mean(y_pred == y))

X_moi = np.array([[1.5, 2.0], [4.5, 2.0]])
print("Dự báo điểm mới:", model.predict(X_moi))