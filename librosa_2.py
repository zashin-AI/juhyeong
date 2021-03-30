# 오디오 특성 추출하기
# import libraries
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# load data

filepath = 'C:/nmb/nmb_data/pansori-tedxkr-corpus-1.0.tar/pansori-tedxkr-corpus-1.0/6rmA6riw7LC9/WRd7fHY3enA/6rmA6riw7LC9-WRd7fHY3enA-0035.flac'

y, sr = librosa.load(
    filepath
)

# Tempo (BPM)
tempo, _ = librosa.beat.beat_track(y, sr = sr)
print(tempo) # 172.265625

# Zero Crossing Rate : 음파가 양에서 음으로 혹은 음에서 양으로 바뀌는 비율
zero_crossings = librosa.zero_crossings(y, pad = False)

print(zero_crossings) # [False False False ... False False False]
print(sum(zero_crossings)) # 5413

n0 = 8000
n1 = 9040

plt.figure(figsize = (16, 6))
plt.plot(y[n0:n1]) # 9000~9040 의 시간대에서 확인함 (제로크로싱이 발생하지 않음) // 8000~9040 인 경우에 제로크로싱이 발생
plt.grid()
plt.show()

# Harmonic and Percussive Components : 사람의 귀로 구분할 수 없는 특징들과 리듬과 감정을 나타내는 충격파
y_harm, y_perc = librosa.effects.hpss(y)

plt.figure(figsize = (16, 6))
plt.plot(y_harm, color = 'b') # 하모닉스는 blue
plt.plot(y_perc, color = 'r') # 퍼커시브는 red
plt.show()

# Spectral Centroid : 소리를 주파수로 표현할 때 가중평균을 계산하여 소리의 무게중심이 어딘지를 알려주는 지표
spectral_centroids = librosa.feature.spectral_centroid(y, sr = sr)[0]

# 시각화를 위한 시간 변수의 연산
frames = range(len(spectral_centroids))

# 음악의 프레임을 초 단위로 변환
t = librosa.frames_to_time(frames)

import sklearn
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

plt.figure(figsize=(16, 6))
librosa.display.waveplot(
    y, sr = sr, alpha = 0.5, color='b'
)
plt.plot(
    t, normalize(spectral_centroids), color = 'r'
)
plt.show()

# Spectral Rolloff : 총 스펙트럼 에너지 중 낮은 주파수(85% 이하) 에 얼마나 많이 집중 되어있는가
spectral_rolloff = librosa.feature.spectral_rolloff(y, sr = sr)[0]

plt.figure(figsize=(16, 6))
librosa.display.waveplot(y, sr = sr, alpha = 0.5, color = 'b')
plt.plot(t, normalize(spectral_rolloff), color = 'r')
plt.show()

# MFCC : 사람의 청각구조를 반영하여 음성 정보를 추출한다 / 작은 특징들의 집합을 요약해서 보여준다
mfccs = librosa.feature.mfcc(y, sr = sr)
mfccs = normalize(mfccs, axis=1)

print(mfccs.mean()) # 0.523752 / 평균값
print(mfccs.var()) # 0.05373637 / 분산값

plt.figure(figsize=(16, 6))
librosa.display.specshow(mfccs, sr = sr, x_axis='time')
plt.show()

# Chroma Frequencies
# 모든 스펙트럼을 12개의 Bin (binary) 로 표현한다 / 12개의 Bin 은 옥타브에서 12개의 각기 다른 반음 (Chroma) 를 의미한다

chromagram = librosa.feature.chroma_stft(y, sr = sr, hop_length=512)

plt.figure(figsize=(16, 6))
librosa.display.specshow(
    chromagram, x_axis='time', y_axis='chroma', hop_length=512
)
plt.show()