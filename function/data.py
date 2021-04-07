import datasave
import numpy as np

filepath = 'C:/nmb/nmb_data/ForM/M/' # 파일 경로
filename = 'flac' # 파일 확장자
labels = 1 # 라벨값 지정

dataset, label = datasave.load_data(filepath, filename, labels)

print(dataset)
print(label)
print(type(dataset))
print(type(label))
