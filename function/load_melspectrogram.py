import os
# import sys
# sys.path.append('c:/nmb/nada/function/')

from feature_handling import load_data_mel

# filepath = 파일 불러올 경로
# filename = 파일 확장자명 (wav, flac...)
# labels = 라벨링 (여자 0, 남자 1)

filepath = 'c:/nmb/nmb_data/ForM/M'
filename = 'flac'
labels = 1

data, label = load_data_mel(filepath, filename, labels)