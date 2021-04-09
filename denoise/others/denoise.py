import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy
import soundfile as sf

import tensorflow

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, \
    Dense, Activation, ReLU, LeakyReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
'''
def make_noise(data, noise_factor):
    noise_datasets = list()
    noise_labels = list()
    for i in data:
        global count
        noise = np.random.randn(len(i))
        data_augment = i + noise * noise_factor
        data_augment = data_augment.astype(type(i[0]))
        noise_datasets.append(data_augment)
        noise_labels.append(0)
        print('noise ' + str(count))

        count += 1
    return noise_datasets, noise_labels

count = 1
dataset = list()
label = list()
filepath = 'c:/nmb/nmb_data/ForM/M/'
filename = 'flac'
labels = 1

files = librosa.util.find_files(filepath, ext=[filename])
files = np.asarray(files)
for file in files:
    y, sr = librosa.load(file, sr=22050, duration=5.0)
    length = (len(y) / sr)
    if length < 5.0 : pass
    else:
        dataset.append(y)
        label.append(labels)
        print(str(count))
        
        count+=1

noise_datasets, noise_labels = make_noise(dataset, 0.01)

print(len(dataset))
print(len(noise_datasets))
print(len(noise_labels))

# a = dataset[0]
# b = noise_datasets[0]

# fig = plt.figure(figsize = (16, 6))
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)

# librosa.display.waveplot(a, ax = ax1)
# librosa.display.waveplot(b, ax = ax2)

# fig.tight_layout()
# plt.show()

dataset = np.array(dataset)
noise_datasets = np.array(noise_datasets)
label = np.array(label)

np.save(
    'c:/nmb/nmb_data/npy/raw_data_male.npy', arr = dataset
)
np.save(
    'c:/nmb/nmb_data/npy/noise_data_male.npy', arr = noise_datasets
)
np.save(
    'c:/nmb/nmb_data/npy/raw_label_male.npy', arr = label
)
'''

