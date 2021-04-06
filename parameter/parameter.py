import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt

# 오디오 파일 로드
y, sr = librosa.load(
    'c:/nmb/nmb_data/주형.wav'
)

print(type(y)) # numpy
print(y.shape) # (110250, )

sm = librosa.feature.stack_memory(
    data = y,
    n_steps = 2,
    delay = 1,
    
)