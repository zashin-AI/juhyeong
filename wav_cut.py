from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPool1D
from tensorflow.keras.models import Sequential

import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# y, sr = librosa.load(
#     'c:/nmb/nmb_data/M5.wav'
# )

# print(len(y))
# print(len(y)/sr)

# length = int(len(y)/sr)
# sec = length * sr

# print(length)
# print(sec)

# y2 = y[:sec]
# # y2 = y[:round(len(y)/sr)]

# print(len(y2))

# sf.write('c:/nmb/nmb_data/M5.wav', y2, sr)

from pydub import AudioSegment

import speech_recognition as sr

y, rate = librosa.load(
    'c:/nmb/nmb_data/STT/STT_F_pred/F_wav/1.wav'
)

sf.write(
    'c:/nmb/nmb_data/STT/STT_F_pred/F_wav/1.wav',
    y, rate
)
print(len(y))

r = sr.Recognizer()

data = sr.AudioFile(
    'c:/nmb/nmb_data/STT/STT_F_pred/F_wav/1.wav'
)