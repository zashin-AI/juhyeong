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
        load_dir : c:/nmb/nmb_data/audio_data/ 로 해주세요
        out_dir : 저장 할 파일 경로
        noise_min : 노이즈 최소값
        noise_max : 노이즈 최대값
        n_fft : n_fft
        hop_length : hop_length
        win_length : win_length

    e.g. :
        denoise_tim(
            'c:/nmb/nmb_data/audio_data/',
            'c:/nmb/nmb_data/audio_data_noise/',
            5000, 15000,
            512, 128, 512
        )
    '''

    for (path, dir, files) in os.walk(load_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            ext_dir = os.path.splitext(path)[0][27:] + '_noise/'
            if ext == '.wav':
                try:
                    if not(os.path.isdir(out_dir + ext_dir)):
                        os.makedirs(os.path.join(out_dir + ext_dir))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        print("Failed to create directory!!!!!")
                        raise
                data, sr = librosa.load("%s/%s" % (path, filename))

                noise_part = data[noise_min:noise_max]

                reduce_noise = nr.reduce_noise(
                    audio_clip=data, 
                    noise_clip=noise_part,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length)

                sf.write(out_dir + ext_dir + filename[:-4] + '_noise.wav', data, sr)
                print("%s/%s" % (path, filename) + ' done')
