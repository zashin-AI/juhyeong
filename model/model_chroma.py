# https://keras.io/examples/audio/speaker_recognition_using_cnn/ 참고 (keras 공식 문서)

import numpy as np
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# 데이터 불러오기
f_ds = np.load('C:/nmb/nmb_data/npy/F_data_chroma_cqt.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/F_label_chroma_cqt.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/M_data_chroma_cqt.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/M_label_chroma_cqt.npy')
# (1073, 128, 862)
# (1073,)

x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
print(x.shape)
print(y.shape)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=45)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 모델 구성
model = Sequential()

def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = Conv1D(filters, 3, padding="same")(x)
        x = Activation(activation)(x)
    x = Conv1D(filters, 3, padding="same")(x)
    x = Add()([x, s])
    x = Activation(activation)(x)
    return MaxPool1D(pool_size=2, strides=1)(x)


def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = AveragePooling1D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inputs, outputs=outputs)

model = build_model(x_train.shape[1:], 2)

model.summary()

# 컴파일, 훈련
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=5, verbose=1)
mcpath = 'C:/nmb/nmb_data/h5/conv1_model_01_cqt.h5'
mc = ModelCheckpoint(mcpath, monitor='val_loss', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, epochs=128, batch_size=32, validation_split=0.2, callbacks=[stop, lr, mc])

# --------------------------------------
# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/conv1_model_01_cqt.h5')

result = model.evaluate(x_test, y_test)
print('loss: ', result[0]); print('acc: ', result[1])

pred_pathAudio = 'C:/nmb/nmb_data/teamvoice_clear/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    cqt = librosa.feature.chroma_cqt(y, sr = sr, hop_length=128)
    # pred_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    pred_cqt = cqt.reshape(1, cqt.shape[0], cqt.shape[1])
    y_pred = model.predict(pred_cqt)
    # print(y_pred)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0 :
        print(file,(y_pred[0][0])*100,'%의 확률로 여자입니다.')
    else: print(file,(y_pred[0][1])*100,'%의 확률로 남자입니다.')

# stft
# loss:  0.44700437784194946
# acc:  0.7627906799316406
# C:\nmb\nmb_data\teamvoice_clear\F1.wav 54.94595170021057 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F1_high.wav 52.23047733306885 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F2.wav 50.01343488693237 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F3.wav 51.353639364242554 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M1.wav 53.14323902130127 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2.wav 50.94479322433472 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2_low.wav 52.138686180114746 %의 확률로 남자입니다.

# cens
# loss:  0.23168054223060608
# acc:  0.930232584476471
# C:\nmb\nmb_data\teamvoice_clear\F1.wav 50.59880018234253 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F1_high.wav 55.33936619758606 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F2.wav 50.24346709251404 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F3.wav 52.01515555381775 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M1.wav 52.100759744644165 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2.wav 53.26981544494629 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2_low.wav 50.38920044898987 %의 확률로 남자입니다.

# cqt
# loss:  0.20572680234909058
# acc:  0.9162790775299072
# C:\nmb\nmb_data\teamvoice_clear\F1.wav 85.13222932815552 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F1_high.wav 67.26455092430115 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F2.wav 87.3017430305481 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F3.wav 89.05304670333862 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M1.wav 84.50077176094055 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2.wav 78.87689471244812 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2_low.wav 86.66472434997559 %의 확률로 남자입니다.