import librosa
import numpy as np
import sklearn

def load_data(filepath, filename):
    count = 1
    dataset = list()
    label = list()
    
    def normalize(x, axis = 0):
        return sklearn.preprocessing.minmax_scale(x, axis = axis)

    files = librosa.util.find_files(filepath, ext=[filename])
    files = np.asarray(files)
    for file in files:
        y, sr = librosa.load(file, sr=22050, duration=5.0)
        length = (len(y) / sr)
        if length < 5.0 : pass
        else:
            mels = librosa.feature.mfcc(y, sr=sr, n_mfcc=20)
            mels = librosa.amplitude_to_db(mels, ref=np.max)
            mels = normalize(mels, axis = 1)

            dataset.append(mels)
            label.append(1)
            print(str(count))
            
            count+=1
    return dataset, label