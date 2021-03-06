import os
import pickle
import numpy as np
import datetime
import librosa
import warnings
warnings.filterwarnings('ignore')
import gc

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler

str_time = datetime.datetime.now()

# data
x = np.load('c:/nmb/nmb_data/npy/total_data.npy')
y = np.load('c:/nmb/nmb_data/npy/total_label.npy')

x = x.reshape(-1, x.shape[1] * x.shape[2])

scale_list = [MinMaxScaler(), StandardScaler()]
# iteration_list = [100, 300, 500, 700, 1000]
iteration_list = [1, 2, 3, 4, 5]
audio_path_list = ['c:/nmb/nmb_data/predict/F', 'c:/nmb/nmb_data/predict/M']

count_f = 0
count_m = 0

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 23
)

for i in scale_list:
    for j in iteration_list:
        scale = i
        scale.fit(x_train)
        x_train = scale.transform(x_train)
        x_test = scale.transform(x_test)

        # for j in model_list:
        model = CatBoostClassifier(
            iterations = j
            )
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_pred)

        print('scaler : \n' + str(i) + '\nmodel : CatBoostClassifier\n' + 'acc : \n', acc)
        print('\nloss : \n', loss)
        pickle.dump(
            model,
            open(
                'c:/data/modelcheckpoint/project_catboost_lr_default_' + str(i) + '_' + str(j) + '.data', 'wb'
            )
        )

        f = open('c:/nmb/nmb_data/' + str(i) + '_' + str(j) + '_' + 'CatBoostClassifier' + '.txt', 'w')
        f.write(
            'scaler : ' + str(i) + '\n' +\
                'iteration : ' + str(j) + '\n' + \
                    'model : CatBoostClassifier' + '\n' +\
                        'acc : ' + str(acc) + '\n' +\
                            'loss : ' + str(loss) + '\n\n')

        for pred in audio_path_list:
            files = librosa.util.find_files(pred, ext = ['wav'])
            files = np.asarray(files)
            
            for file in files:
                name = os.path.basename(file)
                length = len(name)
                name = name[0]

                y, sr = librosa.load(file, sr = 22050)
                mels = librosa.feature.melspectrogram(
                    y, sr = sr, n_fft = 512, hop_length = 128, win_length = 512
                )
                y_mels = librosa.amplitude_to_db(mels ,ref = np.max)
                y_mels = y_mels.reshape(1, y_mels.shape[0] * y_mels.shape[1])

                y_mels = scale.transform(y_mels)

                y_pred = model.predict(y_mels)

                if y_pred == 0:
                    if name == 'F':
                        count_f += 1

                elif y_pred == 1:
                    if name == 'M':
                        count_m += 1

        print('43??? ?????? ????????? ??? ????????? ' + str(count_f) + '??? ?????????.')
        print('43??? ?????? ????????? ??? ????????? ' + str(count_m) + '??? ?????????.')
        print('time : ', datetime.datetime.now() - str_time)
        f.write('43??? ?????? ????????? ??? ????????? ' + str(count_f) + '??? ?????????.\n43??? ?????? ????????? ??? ????????? ' + str(count_m) + '??? ?????????.\n' + \
            'time : ' + str(datetime.datetime.now() - str_time))
        f.close()
        gc.collect()