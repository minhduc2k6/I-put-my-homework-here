def derivative(a, x, n):
    return n * a * x**(n-1)

x=2
tong = 0

for a, n in [(1, 3), (2, 4), (5, 0)]:
    tong += derivative(a, x, n)
print(tong)