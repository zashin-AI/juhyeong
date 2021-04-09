# 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow
import sklearn

from keras.layers import Dense, Conv1D, MaxPool1D,\
    Input, BatchNormalization, Activation
from keras.models import Sequential, Model

# 로드 데이터
# y1, sr1 = librosa.load(
#     'c:/nmb/nmb_data/F1_high.wav'
# ) # 여성 화자

y2, sr2 = librosa.load(
    'c:/nmb/nmb_data/M2_low.wav'
) # 남성 화자

# 필요함수 생성 (노이즈 생성, 정규화)
def noising(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor*noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def normalize(data, axis = 0):
    return sklearn.preprocessing.minmax_scale(data, axis = axis)

# 원본 음성, 노이즈 합성
# noise1 = noising(
#     y1, 0.05
# )

noise2 = noising(
    y2, 0.05
)
# noise2 = np.fft.ifft(noise2).real

plt.figure(figsize = (16, 6))

librosa.display.waveplot(noise2, sr = sr2)
plt.show()