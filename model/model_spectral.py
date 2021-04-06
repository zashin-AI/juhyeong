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
f_ds = np.load('C:/nmb/nmb_data/npy/F_data_spectral_flatness.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/F_label_spectral_flatness.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/M_data_spectral_flatness.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/M_label_spectral_flatness.npy')
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
    return MaxPool1D(pool_size=2, strides=1, padding='same')(x)


def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = AveragePooling1D(pool_size=3, strides=3, padding='same')(x)
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
mcpath = 'C:/nmb/nmb_data/h5/conv1_model_01_spectral_flatness.h5'
mc = ModelCheckpoint(mcpath, monitor='val_loss', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, epochs=128, batch_size=32, validation_split=0.2, callbacks=[stop, lr, mc])

# --------------------------------------
# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/conv1_model_01_spectral_flatness.h5')

result = model.evaluate(x_test, y_test)
print('loss: ', result[0]); print('acc: ', result[1])

pred_pathAudio = 'C:/nmb/nmb_data/teamvoice_clear/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    spectral_flatness = librosa.feature.spectral_flatness(y, n_fft = 512, hop_length=128)
    # pred_spectral_flatness = librosa.amplitude_to_db(spectral_flatness, ref=np.max)
    pred_spectral_flatness = spectral_flatness.reshape(1, spectral_flatness.shape[0], spectral_flatness.shape[1])
    y_pred = model.predict(pred_spectral_flatness)
    # print(y_pred)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0 :
        print(file,(y_pred[0][0])*100,'%의 확률로 여자입니다.')
    else: print(file,(y_pred[0][1])*100,'%의 확률로 남자입니다.')

# spectral bandwidth
# loss:  0.6867212057113647
# acc:  0.5116279125213623
# C:\nmb\nmb_data\teamvoice_clear\F1.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F1_high.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F2.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F3.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M1.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2_low.wav 100.0 %의 확률로 여자입니다.

# spectral centroid
# loss:  0.6931828260421753
# acc:  0.47441861033439636
# C:\nmb\nmb_data\teamvoice_clear\F1.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F1_high.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F2.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F3.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M1.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2_low.wav 100.0 %의 확률로 여자입니다.

# spectral contrast
# loss:  0.00019431315013207495
# acc:  1.0
# C:\nmb\nmb_data\teamvoice_clear\F1.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F1_high.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F2.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F3.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M1.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2.wav 100.0 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2_low.wav 100.0 %의 확률로 여자입니다.

# spectral rolloff
# loss:  0.6925438642501831
# acc:  0.4976744055747986
# C:\nmb\nmb_data\teamvoice_clear\F1.wav 51.729339361190796 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F1_high.wav 51.63537859916687 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F2.wav 51.7463743686676 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F3.wav 51.72698497772217 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M1.wav 51.70819163322449 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2.wav 51.66774392127991 %의 확률로 여자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2_low.wav 51.5142023563385 %의 확률로 여자입니다.

# spectral flatness
# loss:  0.7055891752243042
# acc:  0.4651162922382355
# C:\nmb\nmb_data\teamvoice_clear\F1.wav 52.13229060173035 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F1_high.wav 52.19880938529968 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F2.wav 52.12936997413635 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\F3.wav 52.188658714294434 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M1.wav 52.15936899185181 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2.wav 52.19978094100952 %의 확률로 남자입니다.
# C:\nmb\nmb_data\teamvoice_clear\M2_low.wav 52.40528583526611 %의 확률로 남자입니다.