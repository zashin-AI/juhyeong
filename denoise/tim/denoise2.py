# 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow
import sklearn
import soundfile as sf
import scipy

from keras.layers import Dense, Conv1D, MaxPool1D,\
    Input, BatchNormalization, Activation
from keras.models import Sequential, Model

# 데이터 로드
data, rate = librosa.load(
    'c:/nmb/nmb_data/M2.wav'
) # 여성 화자

# data = data * 10

# y2, sr2 = librosa.load(
#     'c:/nmb/nmb_data/M2_low.wav'
# ) # 남성 화자

data_length = int(len(data)/rate)

# 필요 함수 생성
def noising(data):
    noise_create = np.random.randn(len(data))
    return noise_create

# def fftnoise(data):
#     temp = np.fft.fft(data)
#     return np.fft.ifft(temp).real

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
noise = normalize(noise, axis = 0, num = 10)

noise_clip = noise[:rate * data_length]
audio_clip_band_limited = noise + data

def stft(y, n_fft, hop_length, win_length):
    return librosa.stft(
        y = y, n_fft = n_fft , hop_length = hop_length, win_length = win_length
    ) # stft 함수

def istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length) # istft (inverse) 함수

def amp_to_db(x):
    return librosa.core.amplitude_to_db(
        x, ref = 1.0, amin = 1e-20, top_db = 80.0
    ) # amplitude 를 dB 로 바꿔 시각화하는 데에 용이

def db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref = 1.0) # dB 를 amplitude 로 바꿈

def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq = 2,
    n_grad_time = 4,
    n_fft = 512,
    win_length = 512,
    hop_length = 128,
    n_std_thresh = 1.5,
    prop_decrease = 1.0,
    verbose = False,
    visual = False
):

    noise_stft = stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = amp_to_db(np.abs(noise_stft))
    mean_freq_noise = np.mean(noise_stft_db, axis = 1)
    std_freq_noise = np.std(noise_stft_db, axis = 1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh

    sig_stft = stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = amp_to_db(np.abs(sig_stft))

    mask_gain_dB = np.min(amp_to_db(np.abs(sig_stft)))

    smoothing_filter = np.outer(
        np.concatenate(
            [ 
                np.linspace(0, 1, n_grad_freq + 1, endpoint = False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint = False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis = 0,
    ).T
    
    sig_mask = sig_stft_db < db_thresh
    sig_mask = scipy.signal.fftconvolve(
        sig_mask, smoothing_filter, mode = 'same'
    )
    sig_mask = sig_mask * prop_decrease

    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (db_to_amp(sig_stft_db_masked * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    ))
    
    recoverd_signal = istft(sig_stft_amp, hop_length, win_length)
    recoverd_spec = amp_to_db(
        np.abs(stft(recoverd_signal, n_fft, hop_length, win_length))
    )
    return recoverd_signal

output = removeNoise(
    audio_clip = audio_clip_band_limited,
    noise_clip = noise_clip
)

print(type(output))
print(output)

# save output file to wav
sf.write(
    'c:/nmb/nmb_data/output.wav', output, rate
)
sf.write(
    'c:/nmb/nmb_data/noise_fac.wav', audio_clip_band_limited, rate
)

# visualization
fig = plt.figure(figsize = (16, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

librosa.display.waveplot(
    data, ax = ax1
)
ax1.set(title = 'original')

librosa.display.waveplot(
    audio_clip_band_limited, ax = ax2
)
ax2.set(title = 'noise')

librosa.display.waveplot(
    output, ax = ax3
)
ax3.set(title = 'denoise')

librosa.display.waveplot(
    noise, ax = ax4
)
ax4.set(title = 'noise_factor')

fig.tight_layout()
plt.show()
