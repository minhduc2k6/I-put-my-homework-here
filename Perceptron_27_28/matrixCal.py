import numpy as np


def wtx(w, x):
    """Tính w^T x (tích vô hướng của w và x)."""
    return np.dot(w, x)


def predict(w, x):
    """Nhãn dự đoán = sign(w^T x): +1, -1 (hoặc 0 nếu nằm đúng trên biên)."""
    return np.sign(wtx(w, x))


def is_misclassified(w, x, y):
    """True nếu nhãn dự đoán khác nhãn thực tế y."""
    return predict(w, x) != y


def update(w, x, y):
    """Quy tắc cập nhật PLA cho điểm bị sai: w_mới = w + y*x."""
    return w + y * x