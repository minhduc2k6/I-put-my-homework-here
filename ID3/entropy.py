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
        labels.append(event[-1])  #Lấy giá trị cuối của mỗi event
        #labels.append(event[-1])
    return labels

def entropyCal(labels):
    totalLabels = len(labels)
    uniqueLabels = list(set(labels))
    entroValLabel = 0.0

    for label in uniqueLabels:
        count = labels.count(label) #số số lượng label trong labels
        probability = count / totalLabels
        sigma = -probability * math.log2(probability)
        entroValLabel += sigma
    return entroValLabel

labels = entropyLabel(dataset, attributes)
entroValLabel = entropyCal(labels)
print(entroValLabel)

def attribute_labels(dataset, col_index, target_value):
    attr1 = []
    for event in dataset:
        if event[col_index] == target_value:
            attr1.append(event[-1])  # Append the label of the event if the attribute matches the target value
    return attr1

def entropyAttr(dataset, col_index):
    value_in_col = []
    total_events = len(dataset)
    for event in dataset:
        value_in_col.append(event[col_index])
    uniqueAttr = list(set(value_in_col))

    entroVal = 0.0
    for value in uniqueAttr:
        sub_labels = attribute_labels(dataset, col_index, value)
        branch_entropy = entropyCal(sub_labels)
        probability = len(sub_labels) / total_events
        entroVal += probability * branch_entropy

    return entroVal
entropyAttrVal = entropyAttr(dataset, 0)
print(entropyAttrVal)