import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load(
    'c:/nmb/nmb_data/F1_high.wav'
)

ft = np.fft.fft(y, axis = -1)
ft = abs(ft)      # [0.48021123 1.86881154 1.24656886 ... 2.02027637 1.24656886 1.86881154]
fr = np.fft.fftfreq(len(y), d = 1.0)

print(ft)
print(ft.shape) # len(y) 만큼 나옴

fig = plt.figure(figsize = (16, 6))

plt.plot(fr, ft)
plt.xlim(0, 0.7)
plt.ylim(0, 700)

fig.tight_layout()
plt.show()