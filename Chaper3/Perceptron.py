import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.w = None

    def add_bias(self, X):
        # Thêm cột toàn số 1 vào cuối ma trận X
        so_mau = X.shape[0]
        cot_bias = np.ones((so_mau, 1))
        return np.hstack((X, cot_bias))

    def fit(self, X, y):
        X_bias = self.add_bias(X)
        so_mau = X_bias.shape[0]
        so_cot = X_bias.shape[1]

        self.w = np.zeros(so_cot)

        epoch = 0
        while epoch < self.max_epochs:
            so_loi = 0
            for i in range(so_mau):
                x_i = X_bias[i]
                y_i = y[i]
                diem = np.dot(self.w, x_i)
                if y_i * diem <= 0:
                    # Mẫu bị phân sai, cập nhật w
                    self.w = self.w + self.learning_rate * y_i * x_i
                    so_loi = so_loi + 1
            if so_loi == 0:
                break  # Không còn mẫu sai thì dừng
            epoch = epoch + 1

    def predict(self, X):
        X_bias = self.add_bias(X)
        so_mau = X_bias.shape[0]
        nhan_du_doan = np.zeros(so_mau)
        for i in range(so_mau):
            diem = np.dot(self.w, X_bias[i])
            if diem >= 0:
                nhan_du_doan[i] = 1
            else:
                nhan_du_doan[i] = -1
        return nhan_du_doan

X = np.array([[2, 3]])
y = np.array([1])
model = Perceptron(learning_rate=1, max_epochs=1)
model.w = np.array([-2.0, 1.0, 0.0])
print(model.predict(X))  # [-1.] -> sai