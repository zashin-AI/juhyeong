import librosa
import soundfile as sf
import os
import numpy as np
import noisereduce as nr

def denoise_tim(
    load_dir,
    out_dir,
    noise_min,
    noise_max,
    n_fft,
    hop_length,
    win_length
):

    '''
    Args :
        load_dir : 불러 올 파일 경로
        out_dir : 저장 할 파일 경로
        noise_min : 노이즈 최소값
        noise_max : 노이즈 최대값
        n_fft : n_fft
        hop_length : hop_length
        win_length : win_length
    '''
    
    data, sr = librosa.load(load_dir)

    noise_part = data[noise_min:noise_max]

    reduce_noise = nr.reduce_noise(
        audio_clip=data, 
        noise_clip=noise_part,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length)

    sf.write(out_dir, data, sr)