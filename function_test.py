
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from pydub import AudioSegment
from python_import.voice_handling import voice_sum
from function.feature_handling import load_data_mfcc, load_data_mel

filepath = 'c:/nmb/nmb_data/ForM/F/'
filename = 'flac'
labels = 0

data, label = load_data_mel(
    filepath = filepath,
    filename = filename,
    labels = labels
)

print(type(data))
print(type(label))
print(data.shape)
print(label)