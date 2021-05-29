import numpy as np

list = np.arange(0, 10)
print(list)
print(str(list[0]))

list_2 = []
list_3 = []

txt_2 = ''
txt_3 = ''
for i in range(len(list)):
    if i % 2 == 0:
        list_2.append(i)
    else:
        list_3.append(i)

    txt_2 += str(list_2) + '\n'
    txt_3 += str(list_3) + '\n'

# print(list_2)
# print(list_3)
print('txt_2 : ', txt_2)
print(f'txt_3 : {txt_3}')