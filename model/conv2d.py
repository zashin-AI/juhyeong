import numpy as np
import librosa
import sklearn
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Activation, Flatten, Add, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.callbacks import ModelCheckpoint
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
# f_ds = np.load('C:/nmb/nmb_data/npy/unbalance_female_data.npy')
# m_ds = np.load('C:/nmb/nmb_data/npy/unbalance_male_data.npy')
# f_lb = np.load('C:/nmb/nmb_data/npy/unbalance_female_label.npy')
# m_lb = np.load('C:/nmb/nmb_data/npy/unbalance_male_label.npy')

# x = np.concatenate([f_ds, m_ds], 0)
# y = np.concatenate([f_lb, m_lb], 0)

x = np.load('c:/nmb/nmb_data/npy/unbalance_male_data.npy')
y = np.load('c:/nmb/nmb_data/npy/unbalance_male_label.npy')
print(x.shape, y.shape) # (2141, 128, 862) (2141,)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)
aaa = 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], aaa)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], aaa)
print(x_train.shape, y_train.shape) # (3840, 128, 862, 1) (3840,)
print(x_test.shape, y_test.shape)   # (768, 128, 862, 1) (768,)

# 모델 구성
model = Sequential()
def residual_block(x, filters, conv_num=3, activation='relu'): 
    # Shortcut
    s = Conv2D(filters, 1, padding='same')(x)
    for i in range(conv_num - 1):
        x = Conv2D(filters, 3, padding='same')(x)
        x = Activation(activation)(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = Concatenate(axis=-1)([x, s])
    x = Activation(activation)(x)
    
    return MaxPool2D(pool_size=2, strides=1)(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name='input')
    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 16, 2)
    x = residual_block(x, 8, 3)
    x = AveragePooling2D(pool_size=3, strides=3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    
    return Model(inputs=inputs, outputs=outputs)
model = build_model(x_train.shape[1:], 2)
print(x_train.shape[1:])    # (128, 862, 1)

model.summary()

# 컴파일, 훈련
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', vactor=0.5, patience=10, verbose=1)
path = 'C:/nmb/nmb_data/h5/model_Conv2D_mels2.h5'
mc = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, epochs=300, batch_size=16, validation_split=0.2, callbacks=[es, lr, mc])

# 평가, 예측
model.load_weights('C:/nmb/nmb_data/h5/model_Conv2D_mels2.h5')
result = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", result[0])
print("acc : ", result[1])
pred_pathAudio = 'C:/nmb/nmb_data/predict/'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)

for file in files:   
    y, sr = librosa.load(file, sr=22050) 
    mels = librosa.feature.melspectrogram(y, sr=sr, hop_length=128, n_fft=512)
    pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])
    y_pred = model.predict(pred_mels)
    # print(y_pred)
    y_pred_label = np.argmax(y_pred)
    # print(y_pred_label)
    if y_pred_label == 0 :                   
        print(file,(y_pred[0][0])*100, '%의 확률로 여자입니다.')
    else:                               
        print(file,(y_pred[0][1])*100, '%의 확률로 남자입니다.')
end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # 

'''
in case of Denoise

loss :  0.08901721984148026
acc :  0.9791666865348816
C:\nmb\nmb_data\predict\F1.wav 99.58985447883606 %의 확률로 여자입니다.1
C:\nmb\nmb_data\predict\F10.wav 98.55871200561523 %의 확률로 여자입니다.2
C:\nmb\nmb_data\predict\F11.wav 95.11972069740295 %의 확률로 여자입니다.3
C:\nmb\nmb_data\predict\F12.wav 99.45355653762817 %의 확률로 여자입니다.4
C:\nmb\nmb_data\predict\F13.wav 89.88836407661438 %의 확률로 여자입니다.5
C:\nmb\nmb_data\predict\F14.wav 99.99998807907104 %의 확률로 여자입니다.6
C:\nmb\nmb_data\predict\F15.wav 99.27014112472534 %의 확률로 여자입니다.7
C:\nmb\nmb_data\predict\F16.wav 90.49709439277649 %의 확률로 여자입니다.8
C:\nmb\nmb_data\predict\F17.wav 99.97108578681946 %의 확률로 여자입니다.9
C:\nmb\nmb_data\predict\F18.wav 99.9991774559021 %의 확률로 여자입니다.10
C:\nmb\nmb_data\predict\F19.wav 99.97920393943787 %의 확률로 여자입니다.11
C:\nmb\nmb_data\predict\F1_high.wav 93.00236701965332 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F2.wav 91.07141494750977 %의 확률로 여자입니다.12
C:\nmb\nmb_data\predict\F20.wav 99.99998807907104 %의 확률로 여자입니다.13
C:\nmb\nmb_data\predict\F21.wav 99.99974966049194 %의 확률로 여자입니다.14
C:\nmb\nmb_data\predict\F22.wav 99.90478157997131 %의 확률로 여자입니다.15
C:\nmb\nmb_data\predict\F23.wav 99.99831914901733 %의 확률로 여자입니다.16
C:\nmb\nmb_data\predict\F24.wav 98.81899356842041 %의 확률로 여자입니다.17
C:\nmb\nmb_data\predict\F25.wav 99.94814991950989 %의 확률로 여자입니다.18
C:\nmb\nmb_data\predict\F26.wav 99.99836683273315 %의 확률로 여자입니다.19
C:\nmb\nmb_data\predict\F27.wav 99.92623925209045 %의 확률로 여자입니다.20
C:\nmb\nmb_data\predict\F28.wav 99.63364601135254 %의 확률로 여자입니다.21
C:\nmb\nmb_data\predict\F29.wav 99.99959468841553 %의 확률로 여자입니다.22
C:\nmb\nmb_data\predict\F2_high.wav 71.83866500854492 %의 확률로 여자입니다.23
C:\nmb\nmb_data\predict\F2_low.wav 64.1947090625763 %의 확률로 여자입니다.24
C:\nmb\nmb_data\predict\F3.wav 99.98939037322998 %의 확률로 여자입니다.25
C:\nmb\nmb_data\predict\F30.wav 99.98633861541748 %의 확률로 여자입니다.26
C:\nmb\nmb_data\predict\F31.wav 100.0 %의 확률로 여자입니다.27
C:\nmb\nmb_data\predict\F32.wav 99.98043179512024 %의 확률로 여자입니다.28
C:\nmb\nmb_data\predict\F33.wav 99.97058510780334 %의 확률로 여자입니다.29
C:\nmb\nmb_data\predict\F34.wav 100.0 %의 확률로 여자입니다.30
C:\nmb\nmb_data\predict\F35.wav 91.0903811454773 %의 확률로 여자입니다.31
C:\nmb\nmb_data\predict\F36.wav 97.85534143447876 %의 확률로 여자입니다.32
C:\nmb\nmb_data\predict\F37.wav 99.96306896209717 %의 확률로 여자입니다.33
C:\nmb\nmb_data\predict\F38.wav 99.99934434890747 %의 확률로 여자입니다.34
C:\nmb\nmb_data\predict\F39.wav 99.56693649291992 %의 확률로 여자입니다.35
C:\nmb\nmb_data\predict\F3_high.wav 99.99622106552124 %의 확률로 여자입니다.36
C:\nmb\nmb_data\predict\F4.wav 99.99995231628418 %의 확률로 여자입니다.37
C:\nmb\nmb_data\predict\F40.wav 99.99480247497559 %의 확률로 여자입니다.38
C:\nmb\nmb_data\predict\F41.wav 91.71547293663025 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F42.wav 100.0 %의 확률로 여자입니다.39
C:\nmb\nmb_data\predict\F43.wav 99.9956727027893 %의 확률로 여자입니다.40
C:\nmb\nmb_data\predict\F5.wav 99.99608993530273 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F6.wav 99.99923706054688 %의 확률로 여자입니다.42
C:\nmb\nmb_data\predict\F7.wav 69.10523772239685 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F8.wav 99.99988079071045 %의 확률로 여자입니다.43
C:\nmb\nmb_data\predict\F9.wav 99.62600469589233 %의 확률로 여자입니다.44
C:\nmb\nmb_data\predict\M1.wav 99.93407130241394 %의 확률로 남자입니다.45
C:\nmb\nmb_data\predict\M10.wav 99.96373653411865 %의 확률로 남자입니다.46
C:\nmb\nmb_data\predict\M11.wav 99.99814033508301 %의 확률로 남자입니다.47
C:\nmb\nmb_data\predict\M12.wav 99.95672106742859 %의 확률로 남자입니다.48
C:\nmb\nmb_data\predict\M13.wav 99.99387264251709 %의 확률로 남자입니다.49
C:\nmb\nmb_data\predict\M14.wav 97.55535125732422 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M15.wav 99.94677901268005 %의 확률로 남자입니다.50
C:\nmb\nmb_data\predict\M16.wav 99.99645948410034 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M17.wav 98.1903612613678 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M18.wav 99.97956156730652 %의 확률로 남자입니다.51
C:\nmb\nmb_data\predict\M19.wav 99.9307632446289 %의 확률로 남자입니다.52
C:\nmb\nmb_data\predict\M2.wav 99.65702295303345 %의 확률로 남자입니다.53
C:\nmb\nmb_data\predict\M20.wav 72.0268189907074 %의 확률로 남자입니다.54
C:\nmb\nmb_data\predict\M21.wav 99.86605644226074 %의 확률로 남자입니다.55
C:\nmb\nmb_data\predict\M22.wav 99.96286630630493 %의 확률로 남자입니다.56
C:\nmb\nmb_data\predict\M23.wav 99.83161687850952 %의 확률로 남자입니다.57
C:\nmb\nmb_data\predict\M24.wav 99.60525035858154 %의 확률로 남자입니다.58
C:\nmb\nmb_data\predict\M25.wav 99.9971866607666 %의 확률로 남자입니다.59
C:\nmb\nmb_data\predict\M26.wav 99.98370409011841 %의 확률로 남자입니다.60
C:\nmb\nmb_data\predict\M27.wav 99.24750924110413 %의 확률로 남자입니다.61
C:\nmb\nmb_data\predict\M28.wav 99.91257786750793 %의 확률로 남자입니다.62
C:\nmb\nmb_data\predict\M29.wav 97.85110354423523 %의 확률로 남자입니다.63
C:\nmb\nmb_data\predict\M2_high.wav 71.70652747154236 %의 확률로 남자입니다.64
C:\nmb\nmb_data\predict\M2_low.wav 99.99997615814209 %의 확률로 남자입니다.65
C:\nmb\nmb_data\predict\M3.wav 99.98687505722046 %의 확률로 남자입니다.66
C:\nmb\nmb_data\predict\M30.wav 99.84355568885803 %의 확률로 남자입니다.67
C:\nmb\nmb_data\predict\M31.wav 88.76700401306152 %의 확률로 남자입니다.68
C:\nmb\nmb_data\predict\M32.wav 99.80012774467468 %의 확률로 남자입니다.69
C:\nmb\nmb_data\predict\M33.wav 99.89914298057556 %의 확률로 남자입니다.70
C:\nmb\nmb_data\predict\M34.wav 63.899677991867065 %의 확률로 남자입니다.71
C:\nmb\nmb_data\predict\M35.wav 52.19617486000061 %의 확률로 남자입니다.72
C:\nmb\nmb_data\predict\M36.wav 96.4559257030487 %의 확률로 남자입니다.73
C:\nmb\nmb_data\predict\M37.wav 99.99697208404541 %의 확률로 남자입니다.74
C:\nmb\nmb_data\predict\M38.wav 99.98550415039062 %의 확률로 남자입니다.75
C:\nmb\nmb_data\predict\M39.wav 97.53206372261047 %의 확률로 남자입니다.76
C:\nmb\nmb_data\predict\M4.wav 99.98254179954529 %의 확률로 남자입니다.77
C:\nmb\nmb_data\predict\M40.wav 99.99973773956299 %의 확률로 남자입니다.78
C:\nmb\nmb_data\predict\M41.wav 99.47574138641357 %의 확률로 남자입니다.79
C:\nmb\nmb_data\predict\M42.wav 99.97805953025818 %의 확률로 남자입니다.80
C:\nmb\nmb_data\predict\M43.wav 99.4885504245758 %의 확률로 남자입니다.81
C:\nmb\nmb_data\predict\M5.wav 99.51272010803223 %의 확률로 남자입니다.82
C:\nmb\nmb_data\predict\M5_high.wav 92.23862886428833 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M5_low.wav 99.99973773956299 %의 확률로 남자입니다.83
C:\nmb\nmb_data\predict\M6.wav 99.99990463256836 %의 확률로 남자입니다.84
C:\nmb\nmb_data\predict\M7_high.wav 79.04896140098572 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M7_low.wav 99.9998927116394 %의 확률로 남자입니다.85
C:\nmb\nmb_data\predict\M8.wav 82.38686323165894 %의 확률로 남자입니다.86
C:\nmb\nmb_data\predict\M9.wav 99.99998807907104 %의 확률로 남자입니다.87
time >>  0:03:22.431868


'''