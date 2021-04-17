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
f_ds = np.load('C:/nmb/nmb_data/npy/female_mel_data.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/male_mel_data.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/female_mel_label.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/male_mel_label.npy')

x = np.concatenate([f_ds, m_ds], 0)
y = np.concatenate([f_lb, m_lb], 0)
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


in case of original

loss :  0.2781011760234833
acc :  0.8880208134651184
C:\nmb\nmb_data\predict\F1.wav 99.41044449806213 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F10.wav 56.96932673454285 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F11.wav 51.986223459243774 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F12.wav 66.96658134460449 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F13.wav 63.06905150413513 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F14.wav 99.44629073143005 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F15.wav 95.16807198524475 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F16.wav 64.12866711616516 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F17.wav 96.02131247520447 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F18.wav 70.99584937095642 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F19.wav 97.35345244407654 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F1_high.wav 89.86919522285461 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F2.wav 57.1435809135437 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F20.wav 98.31114411354065 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F21.wav 99.88701939582825 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F22.wav 86.23999953269958 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F23.wav 99.70295429229736 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F24.wav 98.58946800231934 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F25.wav 88.95132541656494 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F26.wav 79.1258156299591 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F27.wav 89.7958517074585 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F28.wav 73.62164855003357 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F29.wav 82.97597169876099 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F2_high.wav 66.05716943740845 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F2_low.wav 50.33271908760071 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F3.wav 97.79434204101562 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F30.wav 97.763192653656 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F31.wav 99.8346209526062 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F32.wav 98.03983569145203 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F33.wav 94.11525130271912 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F34.wav 99.27116632461548 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F35.wav 70.70979475975037 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F36.wav 62.264811992645264 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F37.wav 95.42267322540283 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F38.wav 95.48613429069519 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F39.wav 86.45042181015015 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F3_high.wav 89.37832713127136 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F4.wav 91.18971228599548 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F40.wav 51.36404037475586 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F41.wav 70.5427885055542 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F42.wav 99.98838901519775 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F43.wav 94.607675075531 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F5.wav 82.4754536151886 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F6.wav 94.05233860015869 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F7.wav 99.24989342689514 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F8.wav 94.12248134613037 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\F9.wav 61.12723350524902 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M1.wav 99.09884333610535 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M10.wav 72.05106019973755 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M11.wav 99.01019334793091 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M12.wav 98.79841804504395 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M13.wav 90.11918902397156 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M14.wav 95.6860363483429 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M15.wav 94.45899724960327 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M16.wav 86.87383532524109 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M17.wav 62.594419717788696 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M18.wav 99.26064014434814 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M19.wav 98.12934994697571 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M2.wav 86.51884198188782 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M20.wav 95.5868661403656 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M21.wav 87.8355085849762 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M22.wav 98.05887341499329 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M23.wav 84.12507176399231 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M24.wav 70.59261202812195 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M25.wav 98.77068400382996 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M26.wav 98.7497091293335 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M27.wav 90.91343283653259 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M28.wav 98.43106269836426 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M29.wav 68.57433319091797 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M2_high.wav 70.26362419128418 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M2_low.wav 99.76993203163147 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M3.wav 99.97875094413757 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M30.wav 77.57525444030762 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M31.wav 54.6538770198822 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M32.wav 95.17754316329956 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M33.wav 85.13792157173157 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M34.wav 84.6308171749115 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M35.wav 81.02140426635742 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M36.wav 69.35502290725708 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M37.wav 99.49694275856018 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M38.wav 99.66487288475037 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M39.wav 94.93058919906616 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M4.wav 99.74270462989807 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M40.wav 99.91161227226257 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M41.wav 99.67544674873352 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M42.wav 95.05338072776794 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M43.wav 98.74948859214783 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M5.wav 99.24378991127014 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M5_high.wav 78.03582549095154 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M5_low.wav 98.69369864463806 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M6.wav 99.68565702438354 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M7_high.wav 71.37112021446228 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M7_low.wav 99.89068508148193 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M8.wav 94.33836340904236 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\M9.wav 70.78195810317993 %의 확률로 남자입니다.
time >>  0:10:56.848810

'''