from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPool1D
from tensorflow.keras.models import Sequential

import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load(
    'c:/nmb/nmb_data/testvoice_F1_high.wav'
)

print(len(y))
print(len(y)/sr)

length = int(len(y)/sr)
sec = length * sr

print(length)
print(sec)

y2 = y[:sec]
# y2 = y[:round(len(y)/sr)]

print(len(y2))

sf.write('c:/nmb/nmb_data/F1_high.wav', y2, sr)
