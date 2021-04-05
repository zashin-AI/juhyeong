
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import librosa

y, sr = librosa.load(
    'c:/nmb/nmb_data/M2_low.wav', duration=5.0
)

noising = np.random.normal(0, 0.5)

noise = y + noising

# y = 원본, noise = 노이즈 추가

# 함수 정의
def stft(y, n_fft, hop_length, win_length):
    return librosa.stft(
        y = y, n_fft = n_fft , hop_length = hop_length, win_length = win_length
    )

def istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)

def amp_to_db(x):
    return librosa.core.amplitude_to_db(
        x, ref = 1.0, amin = 1e-20, top_db = 80
    )

def db_to_amp(x,):
    return librosa.cor.db_to_amplitude(x, ref = 1.0)

def removeNoise(
    audio_clip,
    noise_clipe,
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
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concoatenate(
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

    