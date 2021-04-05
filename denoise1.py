# import libraries
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import tensorflow
import sklearn

from keras.layers import Dense, Conv1D, MaxPool1D, Input,\
    BatchNormalization, Activation
from keras.models import Model

# define noise function
def noising(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor*noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def normalize(data, axis = 0):
    return sklearn.preprocessing.minmax_scale(data, axis = axis)

# load wav file
y1, sr1 = librosa.load('c:/nmb/nmb_data/F1_high.wav')
y2, sr2 = librosa.load('c:/nmb/nmb_data/M2_low.wav')

# create noise
noise1 = noising(y1, np.random.normal(0, 0.2))
noise2 = noising(y2, np.random.normal(0, 0.2))

signal_noise1 = y1 + noise1
signal_noise2 = y2 + noise2

# visualization
# fig = plt.figure(figsize = (32, 12))
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)

# librosa.display.waveplot(
#     y, sr = sr, ax = ax1
# )
# ax1.set(title = 'original')

# librosa.display.waveplot(
#     signal_noise, sr = sr, ax = ax2
# )
# ax2.set(title = 'noise')

# fig.tight_layout()
# plt.show()

# save noise file
# sf.write(
#     'c:/nmb/nmb_data/noise.wav', signal_noise, sr
# )

print(y1.shape)
print(y2.shape)
print(signal_noise1.shape)
print(signal_noise2.shape)
'''
stft_1 = np.abs(
    librosa.stft(
        y1,
        n_fft = 512,
        hop_length=128,
        win_length=512
    )
)

stft_2 = np.abs(
    librosa.stft(
        y2,
        n_fft=512,
        hop_length=128,
        win_length=512
    )
)

stft_noise1 = np.abs(
    librosa.stft(
        signal_noise1,
        n_fft=512,
        hop_length=128,
        win_length=512
    )
)

stft_noise2= np.abs(
    librosa.stft(
        signal_noise2,
        n_fft=512,
        hop_length=128,
        win_length=512
    )
)

print(stft_1.shape) # (257, 862)
print(stft_noise1.shape)
'''

mfcc_y1 = librosa.feature.mfcc(
    y1, sr = sr1
)
mfcc_y1 = normalize(mfcc_y1, axis = 1)

mfcc_y2 = librosa.feature.mfcc(
    y2, sr = sr2
)
mfcc_y2 = normalize(mfcc_y2, axis = 1)

mfcc_noise1 = librosa.feature.mfcc(
    signal_noise1, sr = sr1
)
mfcc_noise1 = normalize(mfcc_noise1, axis = 1)

mfcc_noise2 = librosa.feature.mfcc(
    signal_noise2, sr = sr2
)
mfcc_noise2 = normalize(mfcc_noise2, axis = 1)

print(mfcc_y1.shape) # (20, 216)
print(mfcc_y2.shape) # (20, 216)
print(mfcc_noise1.shape) # (20, 216)
print(mfcc_noise2.shape) # (20, 216)

mfcc_y1 = mfcc_y1.reshape(1, 20, 216)
mfcc_y2 = mfcc_y2.reshape(1, 20, 216)
mfcc_noise1 = mfcc_noise1.reshape(1, 20, 216)
mfcc_noise2 = mfcc_noise2.reshape(1, 20, 216)


input1 = Input(shape = (20, 216))
x1 = Dense(1024)(input1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = Dense(1024)(x1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x1 = Dense(1024)(x1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
output1 = Dense(216)(x1)
model1 = Model(input1, output1)

model1.compile(
    optimizer = 'adam', loss = 'mse', metrics= 'mae'
)
model1.fit(
    mfcc_noise1, mfcc_y1,
    epochs = 3, batch_size = 128
)

input2 = Input(shape = (20, 216))
x2 = Conv1D(1024, 3, padding='same')(input2)
x2 = BatchNormalization()(x2)
x2 = MaxPool1D(2, padding='same')(x2)
x2 = Activation('relu')(x2)
x2 = Conv1D(1024, 3, padding='same')(x2)
x2 = BatchNormalization()(x2)
x2 = MaxPool1D(2, padding='same')(x2)
x2 = Activation('relu')(x2)
x2 = Conv1D(1024, 3, padding='same')(x2)
x2 = BatchNormalization()(x2)
x2 = MaxPool1D(2, padding='same')(x2)
x2 = Activation('relu')(x2)
output2 = Dense(216)(x2)
model2 = Model(input2, output2)

model2.compile(
    optimizer = 'adam', loss = 'mse', metrics = 'mae'
)
model2.fit(
    mfcc_noise2, mfcc_y2,
    epochs = 3, batch_size = 128
)
