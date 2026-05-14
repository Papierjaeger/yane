from src.examples import TrainingData

dataset = TrainingData.load_data('PI.json')

PI = TrainingData.get_input_data(dataset, 0)

new_dataset = "["

length = len(PI) - 1

for i in range(length):
    if i == 1:
        continue

    print(f"{i} out of {length} done.", end="\r")

    new_dataset += "{"
    new_dataset += "\"input\": ["
    new_dataset += str(PI[i])
    new_dataset += "],"
    new_dataset += "\"output\": ["

    if i == 0:
        new_dataset += str(PI[i + 2])
    else:
        if i == length:
            new_dataset += str(PI[i])
        else:
            new_dataset += str(PI[i + 1])
    new_dataset += "]"

    if i == length - 1:
        new_dataset += "}"
    else:
        new_dataset += "},"

new_dataset += "]"

with open('dataset_PI.json', 'w') as outfile:
    outfile.write(new_dataset)

print(new_dataset)
