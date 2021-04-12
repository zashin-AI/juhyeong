import noisereduce as nr
import librosa
import soundfile as sf

'''
data, rate = librosa.load(
    'c:/nmb/nmb_data/M5.wav'
)

print(len(data))
print(rate)
print(len(data)/rate)

noise_part = data[10000:15000]

reduce_noise = nr.reduce_noise(
    audio_clip=data, 
    noise_clip=noise_part,
    n_fft=512,
    hop_length=128,
    win_length=512)

sf.write(
    'c:/nmb/nmb_data/reduce_noise_M5.wav', reduce_noise, rate
)
'''
import sys
sys.path.append(
    'c:/nmb/nada/'
)

import function.datasave as ds
data, label = ds.load_data_mfcc(
    'c:/nmb/nmb_data/', 'wav', 1
)
