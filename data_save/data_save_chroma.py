# 데이터 저장
# female = 0, male = 1

import numpy as np
import librosa
import sklearn
import matplotlib.pyplot as plt
import datetime

str_time = datetime.datetime.now()

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

count = 1

dataset = []
label = []
pathAudio = 'C:/nmb/nmb_data/ForM/M/'
files = librosa.util.find_files(pathAudio, ext=['flac'])
files = np.asarray(files)
for file in files:
    y, sr = librosa.load(file, sr=22050, duration=5.0)
    length = (len(y) / sr)
    if length < 5.0 : pass
    else:
        mels = librosa.feature.chroma_cqt(y, sr=sr, hop_length=128)
        mels = librosa.amplitude_to_db(mels, ref=np.max)

        dataset.append(mels)
        label.append(1)
        print(str(count))
        
        count+=1

dataset = np.array(dataset)
label = np.array(label)
print(dataset.shape)
print(label.shape)

np.save('C:/nmb/nmb_data/npy/M_data_chroma_cqt.npy', arr=dataset)
np.save('C:/nmb/nmb_data/npy/M_label_chroma_cqt.npy', arr=label)
print('=====save done=====')
print('time : ', datetime.datetime.now() - str_time)

# done - stft, cqt, cens

# female cens = (545, 12, 862)
# male cens = (528, 12, 862)
# female stft = (545, 12, 862)
# male stft = (528, 12, 862)
# female cqt = (545, 12, 862)
# male cqt = (528, 12, 862) - (12, 862) : 12 - 12 반음, 862 - ?