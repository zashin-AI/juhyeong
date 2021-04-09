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

    # Args:
    #     audio_clip (array): The first parameter. / 기존 음성
    #     noise_clip (array): The second parameter. / 노이즈 음성
    #     n_grad_freq (int): how many frequency channels to smooth over with the mask. / mask 로 처리 할 주파수 채널의 수
    #     n_grad_time (int): how many time channels to smooth over with the mask. / mask 로 부드럽게 처리 할 시간 채널의 수
    #     n_fft (int): number audio of frames between STFT columns. / stft 컬럼을 몇 개로 자를 것인가
    #     win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`.. / window 사이즈
    #     hop_length (int):number audio of frames between STFT columns. / stft 를 자를 때 얼마만큼 겹칠 것인가
    #     n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal / 신호로 간주 할 잡음 수
    #     prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none) / denoise 를 얼마만큼 실행시킬 것인가
    #     visual (bool): Whether to plot the steps of the algorithm / 시각화 관련

    # if verbose:
    noise_stft = stft(noise_clip, n_fft, hop_length, win_length) # noise file 를 받아 stft 화 시킴
    noise_stft_db = amp_to_db(np.abs(noise_stft)) # stft 를 dB 로 바꿔줌
    mean_freq_noise = np.mean(noise_stft_db, axis = 1) # stft 된 noise 의 평균
    std_freq_noise = np.std(noise_stft_db, axis = 1) #  stft 된 noise 의 표준편차
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh # noise stft 의 표준푠차와 n_std_thresh 를 곱한 값에 평균을 더함
    print('noise_stft pass')

    # if verbose:
    sig_stft = stft(audio_clip, n_fft, hop_length, win_length) # 원본 파일을 받아 stft 화 시킴
    sig_stft_db = amp_to_db(np.abs(sig_stft)) # stft 를 dB 로 바꿔줌
    print('sig_stft pass')

    mask_gain_dB = np.min(amp_to_db(np.abs(sig_stft))) # 원본 파일 stft 의 데이터를 dB 한 값의 최소값을 반환

    smoothing_filter = np.outer( # 두 행렬의 곱. 여기선 각각 np.concatenate 들이다.
        np.concatenate( # 1차원 배열들을 concat 함
            [ 
                np.linspace(0, 1, n_grad_freq + 1, endpoint = False),
                # linspace 함수를 이용하여 0 부터 1 까지 n_grad_freq + 1 (==3) 개의 1차원 배열을 생성
                # endpoint = False 이므로 1 은 제외
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
    ).T # .T == np.transpose()
    
    sig_mask = sig_stft_db < db_thresh
    sig_mask = scipy.signal.fftconvolve(
        sig_mask, smoothing_filter, mode = 'same'
    )
    sig_mask = sig_mask * prop_decrease

    # if verbose:
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (db_to_amp(sig_stft_db_masked * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    ))
    print('sig_stft_db pass')
    
    # if verbose:
    recoverd_signal = istft(sig_stft_amp, hop_length, win_length)
    recoverd_spec = amp_to_db(
        np.abs(stft(recoverd_signal, n_fft, hop_length, win_length))
    )
    print('recoverd_signal pass')
    # if verbose:
    print("finish")
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
