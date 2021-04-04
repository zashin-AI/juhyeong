# import libraries
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

y, sr = librosa.load(
    'c:/nmb/nmb_data/주형.wav'
)

print(y)
print(len(y)) # 110250
print(sr) # 22050
print(len(y)/sr) # 5.0 sec
print(y.shape) # (110250, )

noise = np.random.normal(
    0, 0.001, y.shape[0]
)

signal_noise = y+noise

fig = plt.figure(figsize = (32, 12))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

librosa.display.waveplot(
    y, sr = sr, ax = ax1
)
ax1.set(title = 'original')

librosa.display.waveplot(
    signal_noise, sr = sr, ax = ax2
)
ax2.set(title = 'noise')

fig.tight_layout()
plt.show()