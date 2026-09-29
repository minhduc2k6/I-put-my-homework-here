n = int(input("Enter the number of attributes/labels: "))

attributes = []

for i in range(n):
    attr = input(f"Enter attribute/label name {i+1}: ")
    attributes.append(attr)

print(attributes)

attributes = ['weather', 'health', 'label']

dataset = [
    ['sunny', 'good', 1],
    ['rainy', 'bad', 0],
    ['sunny', 'bad', 1],
    ['rainy', 'good', 1]
]