from derivativeCal import grad, cost


def nhap_da_thuc():
    so_luong = int(input("Số hạng tử của f(x): "))
    terms = []
    for i in range(so_luong):
        print(f"Hạng tử {i + 1}:")
        a = float(input("  Hệ số a: "))
        n = int(input("  Số mũ n: "))
        terms.append((a, n))
    return terms


def main():
    terms = nhap_da_thuc()
    x = float(input("Điểm khởi tạo x0: "))
    lr = float(input("Learning rate: "))
    so_buoc = int(input("Số bước cập nhật: "))

    print(f"\nBước 0: x = {x:.4f}, f(x) = {cost(terms, x):.4f}")
    for buoc in range(1, so_buoc + 1):
        g = grad(terms, x)
        x = x - lr * g
        print(f"Bước {buoc}: f'(x_cũ) = {g:.4f}, x = {x:.4f}, f(x) = {cost(terms, x):.4f}")


main()