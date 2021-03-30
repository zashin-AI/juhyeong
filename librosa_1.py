# import libraries
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# load data
filepath = 'C:/nmb/nmb_data/pansori-tedxkr-corpus-1.0.tar/pansori-tedxkr-corpus-1.0/6rmA6riw7LC9/WRd7fHY3enA/6rmA6riw7LC9-WRd7fHY3enA-0035.flac'
filepath2 = 'C:/nmb/nmb_data/pansori-tedxkr-corpus-1.0.tar/pansori-tedxkr-corpus-1.0/6rmA6riw7LC9/WRd7fHY3enA/6rmA6riw7LC9-WRd7fHY3enA-0044.flac'

y, sr = librosa.load(
    filepath
)

print(y) # [-0.00203306 -0.0030984  -0.00270799 ...  0.00321338  0.00340744  0.        ] - 음파의 세기 (진폭)
print(len(y)) # 67253 - 총 음파의 길이
print(sr) # 22050 - 초당 샘플의 갯수 단위는 Hz, kHz
print(len(y)/sr) # 3.05

# visualizaition
plt.figure(figsize = (16, 6))
librosa.display.waveplot(y = y, sr = sr) # wav 파일의 시각화
plt.show() # 가로축은 시간, 세로축은 진폭을 나타낸다

# fourie transformation
D = np.abs(
    librosa.stft(
        y, n_fft = 2048, hop_length = 512
    )
)

print(D.shape) # (1025, 132)

plt.figure(figsize = (16, 6))
plt.plot(D)
plt.show()

# spectogram
DB = librosa.amplitude_to_db(D, ref = np.max)

plt.figure(figsize = (16, 6))
librosa.display.specshow(
    DB, sr = sr, hop_length = 512, x_axis = 'time', y_axis = 'log'
)
plt.colorbar()
plt.show()

# mel spectogram
S = librosa.feature.melspectrogram(y, sr = sr) # y 를 melspectogram 화 시킨다.
S_DB = librosa.amplitude_to_db(S, ref = np.max)

plt.figure(figsize = (16, 6))
librosa.display.specshow(
    S_DB, sr = sr, hop_length = 512, x_axis = 'time', y_axis = 'log'
)
plt.colorbar()
plt.show()

# 다른 음성파일과 Mel Spectogram 을 비교
a, sr = librosa.load(
    filepath
)
a, _ = librosa.effects.trim(a)

A = librosa.feature.melspectrogram(a, sr = sr)
A_DB = librosa.amplitude_to_db(A, ref = np.max)

plt.figure(figsize = (16, 6))
librosa.display.specshow(
    A_DB, sr = sr, hop_length = 512, x_axis = 'time', y_axis = 'log'
)
plt.colorbar()
plt.show()