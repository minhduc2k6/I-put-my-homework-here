import numpy as np
from matrixCal import wtx, predict, is_misclassified, update


def nhap_vector(ten):
    """Nhập 1 vector trên 1 dòng, các số cách nhau bằng dấu cách."""
    return np.array(list(map(float, input(f"Nhập {ten} (cách nhau bằng dấu cách): ").split())))


def main():
    w = nhap_vector("w")
    x = nhap_vector("x (đã thêm bias)")

    if len(w) != len(x):
        print("Lỗi: w và x phải có cùng số phần tử!")
        return

    y = int(input("Nhãn thực tế y (1 hoặc -1): "))

    z = wtx(w, x)
    y_pred = predict(w, x)
    print(f"\n1. w^T x = {z}")
    print(f"2. Nhãn dự đoán = sign({z}) = {int(y_pred)}")

    if is_misclassified(w, x, y):
        print(f"3. Dự đoán {int(y_pred)} khác nhãn thật {y} -> điểm bị phân lớp SAI")
        w_new = update(w, x, y)
        print(f"   w mới = w + y*x = {w_new}")
        print(f"   w_mới^T x = {wtx(w_new, x)}")
    else:
        print(f"3. Dự đoán {int(y_pred)} trùng nhãn thật {y} -> điểm được phân lớp ĐÚNG")


main()