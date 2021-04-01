from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import librosa

audio, sr = librosa.load(
    'c:/data/sushi2.wav'
)

print(audio)
print(len(audio)) # 304820
print(sr) # 22050
print(len(audio)/sr) # 13 sec