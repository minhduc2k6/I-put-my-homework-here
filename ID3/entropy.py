import math
from create_dataset import inputAttributes, inputEvents

print("HELLO AND WELCOME TO MY UNIVERSE")
print("--------GENERATING DATASET--------")
attributes = inputAttributes()
dataset = inputEvents(attributes)

print("--------DATASET GENERATED--------")
print(dataset)

print("--------Calculating Entropy--------")
def entropyLabel(dataset, attributes):
    labels = []
    n = len(attributes)
    for event in dataset:  #for i in range(m) lặp lấy tổng giá trị bằng events
        labels.append(event[n-1])  #Lấy giá trị cuối của mỗi event
    return labels