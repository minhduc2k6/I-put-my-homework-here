def inputAttributes():
    n = int(input("Enter the number of attributes/labels: "))

    attributes = []

    for i in range(n):
        attr = input(f"Enter attribute/label name {i+1}: ")
        attributes.append(attr)

    return attributes

def inputEvents(attributes):
    m = int(input("Enter number of events: "))
    dataset = []
    n = len(attributes)

    for i in range(m):
        print(f"--------Event {i+1}--------")
        event = []
        for j in range(n):
            val = input(f"Enter value for attribute/label '{attributes[j]}': ")
            event.append(val)
        dataset.append(event)
    return dataset

attributes = inputAttributes()
print(attributes)
dataset = inputEvents(attributes)
print(dataset)