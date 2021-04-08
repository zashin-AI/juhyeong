import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load(
    'c:/nmb/nmb_data/F1_high.wav'
)

ft = np.fft.fft(y)
ft = np.abs(ft)

ifft = np.fft.ifft(ft)
ifft = np.abs(ifft)

fig = plt.figure(figsize = (16, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

librosa.display.waveplot(y, sr = sr, ax = ax1)
librosa.display.waveplot(ft, sr = sr, ax = ax2)
librosa.display.waveplot(ifft, sr = sr, ax = ax3)

fig.tight_layout()
plt.show()