import os
import numpy as np
import datetime
import librosa
import pickle
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tensorflow.keras.models import load_model

str_time = datetime.datetime.now()

x = np.load('c:/nmb/nmb_data/npy/total_data.npy')
y = np.load('c:/nmb/nmb_data/npy/total_label.npy')

x = x.reshape(-1, x.shape[1] * x.shape[2])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 23
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# model
model = CatBoostClassifier(
    # learning_rate=0.01,
    iterations=1000
)
model.fit(x_train, y_train)

# 가중치 저장
pickle.dump(
    model,
    open(
        'c:/data/modelcheckpoint/project_catboost__iter_1000_ss.data', 'wb')
    )

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_pred)

print('acc : ', acc)
print('loss : ', loss)

# predict
pred_list = ['c:/nmb/nmb_data/predict/F', 'c:/nmb/nmb_data/predict/M']

count_f = 0
count_m = 0

for pred in pred_list:
    files = librosa.util.find_files(pred, ext = ['wav'])
    files = np.asarray(files)
    for file in files:
        name = os.path.basename(file)
        length = len(name)
        name = name[0]

        y, sr = librosa.load(file, sr = 22050)
        mels = librosa.feature.melspectrogram(
            y, sr = sr, n_fft = 512, hop_length=128, win_length=512
        )
        y_mels = librosa.amplitude_to_db(mels, ref = np.max)
        y_mels = y_mels.reshape(1, y_mels.shape[0] * y_mels.shape[1])

        y_mels = scaler.transform(y_mels)

        y_pred = model.predict(y_mels)

        if y_pred == 0:
            if name == 'F':
                count_f += 1
        elif y_pred == 1:
            if name == 'M':
                count_m += 1

print('43개의 목소리 중 여자는 ' + str(count_f) + ' 개 입니다.')
print('43개의 목소리 중 남자는 ' + str(count_m) + ' 개 입니다.')
print('time : ', datetime.datetime.now() - str_time)

# lr = 0.017861
# acc :  0.920704845814978
# loss :  2.7387848986276495
# 43개의 목소리 중 여자는 38 개 입니다.
# 43개의 목소리 중 남자는 39 개 입니다.
# time :  1:16:38.881588