import numpy as np
import librosa

data, rate = librosa.load('c:/nmb/nmb_data/M5.wav')

a = np.linspace(0, 1, 3, endpoint=False)
b = np.linspace(1, 0, 4, endpoint=True)
# print(a)
# print(b)

a1 = np.linspace(0, 1, 5, endpoint=False)
b1 = np.linspace(1, 0, 6, endpoint=True)

c = np.concatenate([a, b,])
# print(c)
c1 = np.concatenate([a1, b1])

d = c[1:-1] # 양 끝에있는 0 을 제외
# print(d)
d1 = c1[1:-1]
# e = c[1:-1]
# e1 = c1[1:-1]
# print(d1)

f = np.outer(d, d1)
print('outer : \n', f)

g = np.sum(f) # 15
f = f / np.sum(f) # 각 인자 별로 15 를 나눔

print('f : \n', f)
print('sum : ', g)

import scipy.fft
import sklearn


def noising(data):
    noise_create = np.random.randn(len(data))
    return noise_create
def normalize(data, axis = 0, num = 1):
    return sklearn.preprocessing.minmax_scale(data, axis = axis) * num

# def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
#     freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
#     f = np.zeros(samples)
#     f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
#     return noising(data)

noise = noising(data)
# noise = fftnoise(noise)
# noise = band_limited_noise(4000, 12000, samples = len(data), samplerate = rate) * 10
noise = normalize(noise, axis = 0, num = 0.1)

data_length = len(data)//rate
noise_clip = noise[:rate * data_length]
audio_clip_band_limited = noise + data

data = librosa.stft(data, rate)
data = librosa.core.amplitude_to_db(data, ref = 1.0)

noise_stft = librosa.stft(noise, rate)
noise_stft = librosa.core.amplitude_to_db(noise_stft, ref = 1.0)

mean_noise_stft = np.mean(noise_stft, axis=1)
std_noise_stft = np.std(noise_stft, axis = 1)

noise_thresh = mean_noise_stft + std_noise_stft * 1.5

db_thresh = np.repeat(
    np.reshape(noise_thresh, [1, len(mean_noise_stft)]),
    np.shape(data)[1],
    axis = 0,
).T

sig_mask = data < db_thresh
sig_mask = scipy.signal.fftconvolve(sig_mask, f, mode = 'same')

import librosa.display
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (16, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

librosa.display.specshow(f, sr = rate, ax = ax1)
librosa.display.specshow(db_thresh, sr = rate, ax = ax2)
librosa.display.waveplot(data, sr = rate, ax = ax3)
librosa.display.waveplot(audio_clip_band_limited, sr = rate, ax = ax4)

ax1.set(title = 'sigmask')
ax2.set(title = 'std')
ax3.set(title = 'noise')
ax4.set(title = 'data with noise')

fig.tight_layout()

plt.show()