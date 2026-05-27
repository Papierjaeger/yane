new_dataset = "["

x = 10
y = 10

for i in range(x):
    for j in range(y):

        print(f"{i * j} out of {x * y} done.", end="\r")

        new_dataset += "{"
        new_dataset += "\"input\": ["
        new_dataset += str(f"{i}, {j}")
        new_dataset += "],"
        new_dataset += "\"output\": ["

        new_dataset += str(f"{i * j}")

        new_dataset += "]"

        if i == x - 1 and j == y - 1:
            new_dataset += "}"
        else:
            new_dataset += "},"

new_dataset += "]"

with open('multiplication_table.json', 'w') as outfile:
    outfile.write(new_dataset)

print(new_dataset)
