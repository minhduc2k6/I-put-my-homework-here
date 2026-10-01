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
        #labels.append(event[-1])
    return labels

def entropyCal(labels):
    totalLabels = len(labels)
    uniqueLabels = list(set(labels))
    entroVal = 0.0

    for label in uniqueLabels:
        count = labels.count(label) #số số lượng label trong labels
        probability = count / totalLabels
        sigma = -probability * math.log2(probability)
        entroVal += sigma
    return entroVal

def