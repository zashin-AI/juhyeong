import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv1D, MaxPooling1D,\
    Dense, LSTM, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential

y, sr = librosa.load(
    'c:/nmb/nmb_data/주형.wav'
)

print(y)
print(len(y)) # 110250
print(sr) # 22050
print(len(y)/sr) # 5.0

S = librosa.feature.melspectrogram(
    y, sr
)
S_DB = librosa.amplitude_to_db(
    S, ref = np.min
)

print(type(S)) # numpy
print(type(S_DB)) # numpy

print(S.shape) # (128, 216)
print(S_DB.shape) # (128, 216)

plt.figure(figsize = (16, 6))
librosa.display.specshow(
    S_DB, sr = sr
)
plt.colorbar()
plt.show()

# S = S.reshape(128, 216, 1)
# S_DB = S_DB.reshape(128, 216, 1)

# model=Sequential()
# model.add(Conv1D(128, 2, padding='same', input_shape = (128, 216)))
# model.add(Conv1D(128, 2, padding='same'))
# model.add(MaxPooling1D(3, padding='same'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()

# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy'
# )

# model.fit(
#     S,
#     epochs=1
# )