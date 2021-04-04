# import libraries
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

# define noise function
def noising(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor*noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

# load wav file
y, sr = librosa.load(
    'c:/nmb/nmb_data/주형.wav'
)

print(y)
print(len(y)) # 110250
print(sr) # 22050
print(len(y)/sr) # 5.0 sec
print(y.shape) # (110250, )

# create noise
noise = noising(y, np.random.normal(0, 0.2))

signal_noise = y+noise

# visualization
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

# save noise file
sf.write(
    'c:/nmb/nmb_data/noise.wav', signal_noise, sr
)