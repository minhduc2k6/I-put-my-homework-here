# 3_26.py
def f(x):
    return x ** 2 - 4 * x + 5


def df(x):
    # Đạo hàm f'(x) = 2x - 4
    return 2 * x - 4


x = 5.0
learning_rate = 0.2

print("Ban dau: x =", x, ", f(x) =", f(x))
print("Buoc | x_cu | f'(x_cu) | x_moi | f(x_moi)")

for buoc in range(4):
    x_cu = x
    dao_ham = df(x_cu)
    x_moi = x_cu - learning_rate * dao_ham
    gia_tri_f = f(x_moi)

    print(buoc, "|", round(x_cu, 4), "|", round(dao_ham, 4), "|",
          round(x_moi, 4), "|", round(gia_tri_f, 4))

    x = x_moi

print("Cuc tieu ly thuyet: x = 2, f(2) =", f(2))