from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load(
    'c:/nmb/nmb_data/testvoice_주형.wav'
)

S = librosa.feature.melspectrogram(y, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref = np.max)

plt.figure(figsize = (16, 6))
librosa.display.specshow(
    S_DB, x_axis = 'time', y_axis = 'log', sr = sr
)
plt.colorbar()
plt.show()