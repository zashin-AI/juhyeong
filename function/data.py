import datasave
import numpy as np

filepath = 'C:/nmb/nmb_data/ForM/M/'
filename = 'flac'

dataset, label = datasave.load_data(filepath, filename, 1)

print(dataset)
print(label)
print(type(dataset))
print(type(label))
