import noisereduce as nr
import librosa
import librosa.display
from numpy import lib
import soundfile as sf
import matplotlib.pyplot as plt

data, rate = librosa.load(
    'c:/nmb/nmb_data/M5.wav'
)

print(len(data))
print(rate)
print(len(data)/rate)

noise_part = data[4000:15000]

reduce_noise = nr.reduce_noise(
    audio_clip=data, 
    noise_clip=noise_part,
    n_fft=512,
    hop_length=128,
    win_length=512)

# sf.write(
#     'c:/nmb/nmb_data/reduce_noise_M5.wav', reduce_noise, rate
# )

fig = plt.figure(figsize = (16, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

librosa.display.waveplot(data, sr = rate, ax = ax1)
librosa.display.waveplot(noise_part, sr = rate, ax = ax2)
librosa.display.waveplot(reduce_noise, sr = rate, ax = ax3)

fig.tight_layout()
plt.show()
