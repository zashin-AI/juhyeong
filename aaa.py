import numpy as np

list = np.arange(0, 10)
print(list)
print(str(list[0]))

list_2 = []
list_3 = []
for i in range(len(list)):
    if i % 2 == 0:
        list_2.append(i)
    else:
        list_3.append(i)

print(list_2)
print(list_3)