"""
Dự đoán giá nhà bằng Hồi quy tuyến tính nhiều biến (diện tích + số phòng tắm)
-------------------------------------------------------------------------------
KHÔNG cần cài thư viện ngoài (pandas/numpy/scikit-learn) — chỉ dùng
thư viện chuẩn của Python (csv). Chạy được ngay bằng lệnh:

    python simple_linear_regression.py

Công thức hồi quy tuyến tính nhiều biến:
    price = b0 + b1 * area_m2 + b2 * bathrooms

Cách giải: dùng phương trình chuẩn (normal equation) của bình phương tối
thiểu, giải hệ phương trình tuyến tính bằng phép khử Gauss viết tay
(không dùng numpy).
"""

import csv

DATA_FILE = "data/house_prices_simple.csv"
FEATURES = ["area_m2", "bathrooms"]
TARGET = "price_million_vnd"


def load_data(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                ([1.0] + [float(row[feat]) for feat in FEATURES], float(row[TARGET]))
            )
    return rows  # list of (x_vector_with_bias, y)


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
            for i in range(len(A))]


def transpose(A):
    return [list(row) for row in zip(*A)]


def solve_linear_system(A, b):
    """Giải Ax = b bằng khử Gauss (A vuông, không dùng thư viện ngoài)."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot_row] = M[pivot_row], M[col]
        pivot = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivot
        for r in range(n):
            if r != col:
                factor = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]

    return [M[i][n] for i in range(n)]


def train_multiple_linear_regression(rows):
    X = [x for x, _ in rows]
    y = [[yi] for _, yi in rows]

    Xt = transpose(X)
    XtX = matmul(Xt, X)
    Xty = matmul(Xt, y)
    Xty_flat = [row[0] for row in Xty]

    coeffs = solve_linear_system(XtX, Xty_flat)
    return coeffs  # [b0, b1, b2, ...] theo đúng thứ tự FEATURES


def predict(coeffs, feature_values):
    return coeffs[0] + sum(c * v for c, v in zip(coeffs[1:], feature_values))


def r_squared(rows, coeffs):
    y_vals = [y for _, y in rows]
    y_mean = sum(y_vals) / len(y_vals)
    ss_res = sum((y - predict(coeffs, x[1:])) ** 2 for x, y in rows)
    ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
    return 1 - ss_res / ss_tot


def main():
    rows = load_data(DATA_FILE)
    print(f"Đã đọc {len(rows)} mẫu dữ liệu từ {DATA_FILE}\n")

    coeffs = train_multiple_linear_regression(rows)
    r2 = r_squared(rows, coeffs)

    b0, b1, b2 = coeffs
    print("=== Kết quả huấn luyện mô hình ===")
    print(f"Công thức: price = {b0:.2f} + {b1:.2f} * area_m2 + {b2:.2f} * bathrooms")
    print(f"R^2 (độ khớp của mô hình): {r2:.4f}\n")

    print("=== Thử dự đoán giá cho vài trường hợp mới ===")
    test_cases = [(50, 1), (70, 2), (100, 2), (120, 3), (150, 3)]
    for area, bathrooms in test_cases:
        gia = predict(coeffs, [area, bathrooms])
        print(f"Nhà {area} m2, {bathrooms} phòng tắm  ->  giá dự đoán ~ {gia:,.0f} triệu VNĐ")


if __name__ == "__main__":
    main()
