import numpy as np


class Perceptron:
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter   # số lượt duyệt dữ liệu tối đa
        self.w = None              # trọng số, chưa có vì chưa học
        self.n_updates = 0         # đếm số lần cập nhật w

    def _add_bias(self, X):
        """Thêm cột số 1 vào đầu mỗi hàng: [x1, x2] -> [1, x1, x2]."""
        return np.column_stack((np.ones(len(X)), X))

    def fit(self, X, y):
        """Học w. X có dạng (N, d): mỗi hàng là một điểm. y gồm các nhãn +1/-1."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        Xb = self._add_bias(X)

        self.w = np.zeros(Xb.shape[1])   # khởi tạo w = 0
        self.n_updates = 0

        for _ in range(self.max_iter):
            so_loi = 0
            for xi, yi in zip(Xb, y):
                if yi * np.dot(self.w, xi) <= 0:      # điểm bị phân lớp sai
                    self.w = self.w + yi * xi          # w_mới = w + y*x
                    self.n_updates += 1
                    so_loi += 1
            if so_loi == 0:                            # cả lượt không sai điểm nào
                break
        return self

    def predict(self, X):
        """Dự báo nhãn +1/-1 cho dữ liệu mới, dạng (N, d)."""
        X = np.asarray(X, dtype=float)
        z = np.dot(self._add_bias(X), self.w)
        return np.where(z >= 0, 1, -1)