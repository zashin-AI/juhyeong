# Steps of algorithm

# 1. An FFT is calculated over the noise audio clip
# 2. Statistics are calculated over FFT of the the noise (in frequency)
# 3. A threshold is calculated based upon the statistics of the noise (and the desired sensitivity of the algorithm)
# 4. An FFT is calculated over the signal
# 5. A mask is determined by comparing the signal FFT to the threshold
# 6. The mask is smoothed with a filter over frequency and time
# 7. The mask is appled to the FFT of the signal, and is inverted


import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import librosa

# 예제 파일 로드
filepath = 'c:/nmb/nmb_data/M2_low.wav'
data, rate = librosa.load(
    filepath
)

print(data)

def fftnoise(f):
    f = np.array(f, dtype = 'complex') # 복수소형의 array 생성
    Np = (len(f) - 1)//2 # array 를 2 로 나눈 후 int 값만 가져옴
    phase = np.random.rand(Np) * 2 * np.pi
    phase = np.cos(phase) + 1j * np.sin(phase)
    f[1 : Np + 1] *= phase
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1]) #켤레 복소수 생성
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples = 1024, samplerate = 1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    return fftnoise(f)

noise_len = 5 # second
noise = band_limited_noise(
    min_freq = 4000, 
    max_freq = 12000, 
    samples = len(data),
    samplerate = rate) * 10
noise_clip = noise[:rate * noise_len] # data 값의 길이를 구하기 위함
audio_clip_band_limited = data + noise

# 함수 정의
def stft(y, n_fft, hop_length, win_length):
    return librosa.stft(
        y = y, n_fft = n_fft , hop_length = hop_length, win_length = win_length
    ) # stft 함수

def istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length) # istft (inverse) 함수

def amp_to_db(x):
    return librosa.core.amplitude_to_db(
        x, ref = 1.0, amin = 1e-20, top_db = 80
    ) # amplitude 를 dB 로 바꿔 시각화하는 데에 용이

def db_to_amp(x,):
    return librosa.cor.db_to_amplitude(x, ref = 1.0) # dB 를 amplitude 로 바꿈

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
    if verbose:
        noise_stft = stft(noise_clip, n_fft, hop_length, win_length)
        noise_stft_db = amp_to_db(np.abs(noise_stft))
        mean_freq_noise = np.mean(noise_stft_db, axis = 1)
        std_freq_noise = np.std(noise_stft_db, axis = 1)
        noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    
    if verbose:
        sig_stft = stft(audio_clip, n_fft, hop_length, win_length)
        sig_stft_db = amp_to_db(np.abs(sig_stft))

    mask_gain_dB = np.min(amp_to_db(np.abs(sig_stft)))

    smooting_filter = np.outer(
        np.concatenate(
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
    ).T
    
    sig_mask = sig_stft_db < db_thresh

    if verbose:
        sig_stft_db_masked = (
            sig_stft_db * (1 - sig_mask)
            + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB
        )
        sig_img_masked = np.imag(sig_stft) * (1 - sig_mask)
        sig_stft_amp = (db_to_amp(sig_stft_db_masked * np.sing(sig_stft)) + (
            1j * sig_img_masked
        ))
    
    if verbose:
        recoverd_signal = istft(sig_stft_amp, hop_length, win_length)
        recoverd_spec = amp_to_db(
            np.abs(stft(recoverd_signal, n_fft, hop_length, win_length))
        )
    if verbose:
        print("finish")

output = removeNoise(
    audio_clip = audio_clip_band_limited,
    noise_clip = noise_clip,
    verbose = True 
)
