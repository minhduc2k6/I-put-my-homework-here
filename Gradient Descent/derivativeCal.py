def derivativeCal(a, x, n):
    """Đạo hàm của một hạng tử a*x^n tại điểm x."""
    if n == 0:          # hằng số -> đạo hàm = 0
        return 0
    return a * n * x**(n - 1)


def grad(terms, x):
    """Đạo hàm của cả đa thức (list các cặp (a, n)) tại điểm x."""
    tong = 0
    for a, n in terms:
        tong += derivativeCal(a, x, n)
    return tong


def cost(terms, x):
    """Giá trị của hàm f(x) = tổng các a*x^n."""
    tong = 0
    for a, n in terms:
        tong += a * x**n
    return tong