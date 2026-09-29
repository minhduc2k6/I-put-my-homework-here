import random
import csv

random.seed(42)
n = 20

rows = []
for _ in range(n):
    area = round(random.uniform(45, 150), 1)
    bathrooms = random.randint(1, 3)
    price = 30 * area + 150 * bathrooms + random.uniform(-120, 120)
    rows.append((area, bathrooms, round(price)))

rows.sort(key=lambda r: r[0])

with open("data/house_prices_simple.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["area_m2", "bathrooms", "price_million_vnd"])
    writer.writerows(rows)

print("Đã tạo lại data/house_prices_simple.csv")
for r in rows:
    print(r)
