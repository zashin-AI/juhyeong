import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr

from tensorflow.keras.layers import Conv1D, MaxPooling1D,\
    Dense, LSTM, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential