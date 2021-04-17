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
f_ds = np.load('C:/nmb/nmb_data/npy/denoise_female_mel_data.npy')
m_ds = np.load('C:/nmb/nmb_data/npy/denoise_male_mel_data.npy')
f_lb = np.load('C:/nmb/nmb_data/npy/denoise_female_mel_label.npy')
m_lb = np.load('C:/nmb/nmb_data/npy/denoise_male_mel_label.npy')

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

loss :  0.2517814338207245
acc :  0.90625
C:\nmb\nmb_data\predict\F1.wav 99.84456896781921 %의 확률로 여자입니다.1
C:\nmb\nmb_data\predict\F10.wav 96.70400023460388 %의 확률로 여자입니다.2
C:\nmb\nmb_data\predict\F11.wav 96.75897359848022 %의 확률로 여자입니다.3
C:\nmb\nmb_data\predict\F12.wav 71.22766971588135 %의 확률로 여자입니다.4
C:\nmb\nmb_data\predict\F13.wav 91.54559969902039 %의 확률로 여자입니다.5
C:\nmb\nmb_data\predict\F14.wav 98.78503680229187 %의 확률로 여자입니다.6
C:\nmb\nmb_data\predict\F15.wav 98.07788133621216 %의 확률로 여자입니다.7
C:\nmb\nmb_data\predict\F16.wav 86.56070828437805 %의 확률로 여자입니다.8
C:\nmb\nmb_data\predict\F17.wav 99.93855953216553 %의 확률로 여자입니다.9
C:\nmb\nmb_data\predict\F18.wav 99.61373209953308 %의 확률로 여자입니다.10
C:\nmb\nmb_data\predict\F19.wav 74.17088150978088 %의 확률로 여자입니다.11
C:\nmb\nmb_data\predict\F1_high.wav 91.6172444820404 %의 확률로 여자입니다.12
C:\nmb\nmb_data\predict\F2.wav 99.09282326698303 %의 확률로 여자입니다.13
C:\nmb\nmb_data\predict\F20.wav 99.99803304672241 %의 확률로 여자입니다.14
C:\nmb\nmb_data\predict\F21.wav 99.92648959159851 %의 확률로 여자입니다.15
C:\nmb\nmb_data\predict\F22.wav 96.51461243629456 %의 확률로 여자입니다.16
C:\nmb\nmb_data\predict\F23.wav 99.81281161308289 %의 확률로 여자입니다.17
C:\nmb\nmb_data\predict\F24.wav 99.51726794242859 %의 확률로 여자입니다.18
C:\nmb\nmb_data\predict\F25.wav 89.36301469802856 %의 확률로 여자입니다.19
C:\nmb\nmb_data\predict\F26.wav 99.50719475746155 %의 확률로 여자입니다.20
C:\nmb\nmb_data\predict\F27.wav 98.795485496521 %의 확률로 여자입니다.21
C:\nmb\nmb_data\predict\F28.wav 99.18545484542847 %의 확률로 여자입니다.22
C:\nmb\nmb_data\predict\F29.wav 97.26799726486206 %의 확률로 여자입니다.23
C:\nmb\nmb_data\predict\F2_high.wav 65.14520049095154 %의 확률로 여자입니다.24
C:\nmb\nmb_data\predict\F2_low.wav 62.93884515762329 %의 확률로 여자입니다.25
C:\nmb\nmb_data\predict\F3.wav 99.94506239891052 %의 확률로 여자입니다.26
C:\nmb\nmb_data\predict\F30.wav 97.69983887672424 %의 확률로 여자입니다.27
C:\nmb\nmb_data\predict\F31.wav 99.99815225601196 %의 확률로 여자입니다.28
C:\nmb\nmb_data\predict\F32.wav 99.99229907989502 %의 확률로 여자입니다.29
C:\nmb\nmb_data\predict\F33.wav 93.9888596534729 %의 확률로 여자입니다.30
C:\nmb\nmb_data\predict\F34.wav 99.9634861946106 %의 확률로 여자입니다.31
C:\nmb\nmb_data\predict\F35.wav 98.97712469100952 %의 확률로 여자입니다.32
C:\nmb\nmb_data\predict\F36.wav 97.93319702148438 %의 확률로 여자입니다.33
C:\nmb\nmb_data\predict\F37.wav 99.20933842658997 %의 확률로 여자입니다.34
C:\nmb\nmb_data\predict\F38.wav 98.37892055511475 %의 확률로 여자입니다.35
C:\nmb\nmb_data\predict\F39.wav 96.00462913513184 %의 확률로 여자입니다.36
C:\nmb\nmb_data\predict\F3_high.wav 98.51493835449219 %의 확률로 여자입니다.37
C:\nmb\nmb_data\predict\F4.wav 98.46967458724976 %의 확률로 여자입니다.38
C:\nmb\nmb_data\predict\F40.wav 99.17415380477905 %의 확률로 여자입니다.39
C:\nmb\nmb_data\predict\F41.wav 96.59912586212158 %의 확률로 여자입니다.40
C:\nmb\nmb_data\predict\F42.wav 99.98229146003723 %의 확률로 여자입니다.41
C:\nmb\nmb_data\predict\F43.wav 99.80477094650269 %의 확률로 여자입니다.42
C:\nmb\nmb_data\predict\F5.wav 99.06368851661682 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F6.wav 99.99914169311523 %의 확률로 여자입니다.43
C:\nmb\nmb_data\predict\F7.wav 92.30291247367859 %의 확률로 여자입니다.44
C:\nmb\nmb_data\predict\F8.wav 99.91201758384705 %의 확률로 여자입니다.45
C:\nmb\nmb_data\predict\F9.wav 73.55250120162964 %의 확률로 여자입니다.46
C:\nmb\nmb_data\predict\M1.wav 96.36598825454712 %의 확률로 남자입니다.47
C:\nmb\nmb_data\predict\M10.wav 73.55760931968689 %의 확률로 남자입니다.48
C:\nmb\nmb_data\predict\M11.wav 94.6790337562561 %의 확률로 남자입니다.49
C:\nmb\nmb_data\predict\M12.wav 98.47395420074463 %의 확률로 남자입니다.50
C:\nmb\nmb_data\predict\M13.wav 92.1143114566803 %의 확률로 남자입니다.51
C:\nmb\nmb_data\predict\M14.wav 99.90826845169067 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M15.wav 98.50409626960754 %의 확률로 남자입니다.52
C:\nmb\nmb_data\predict\M16.wav 98.00467491149902 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M17.wav 96.2235689163208 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M18.wav 99.58027005195618 %의 확률로 남자입니다.53
C:\nmb\nmb_data\predict\M19.wav 96.88766598701477 %의 확률로 남자입니다.54
C:\nmb\nmb_data\predict\M2.wav 92.06618666648865 %의 확률로 남자입니다.55
C:\nmb\nmb_data\predict\M20.wav 90.84299802780151 %의 확률로 남자입니다.56
C:\nmb\nmb_data\predict\M21.wav 80.58555126190186 %의 확률로 남자입니다.57
C:\nmb\nmb_data\predict\M22.wav 98.41073751449585 %의 확률로 남자입니다.58
C:\nmb\nmb_data\predict\M23.wav 86.6247832775116 %의 확률로 남자입니다.59
C:\nmb\nmb_data\predict\M24.wav 86.81851625442505 %의 확률로 남자입니다.60
C:\nmb\nmb_data\predict\M25.wav 98.57640862464905 %의 확률로 남자입니다.61
C:\nmb\nmb_data\predict\M26.wav 97.35852479934692 %의 확률로 남자입니다.62
C:\nmb\nmb_data\predict\M27.wav 85.58233976364136 %의 확률로 남자입니다.63
C:\nmb\nmb_data\predict\M28.wav 97.85299897193909 %의 확률로 남자입니다.64
C:\nmb\nmb_data\predict\M29.wav 77.33826637268066 %의 확률로 남자입니다.65
C:\nmb\nmb_data\predict\M2_high.wav 93.28792691230774 %의 확률로 남자입니다.66
C:\nmb\nmb_data\predict\M2_low.wav 99.87000226974487 %의 확률로 남자입니다.67
C:\nmb\nmb_data\predict\M3.wav 99.85672235488892 %의 확률로 남자입니다.68
C:\nmb\nmb_data\predict\M30.wav 80.29589653015137 %의 확률로 남자입니다.69
C:\nmb\nmb_data\predict\M31.wav 72.68422245979309 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M32.wav 99.08684492111206 %의 확률로 남자입니다.70
C:\nmb\nmb_data\predict\M33.wav 99.16847944259644 %의 확률로 남자입니다.71
C:\nmb\nmb_data\predict\M34.wav 84.1207504272461 %의 확률로 남자입니다.72
C:\nmb\nmb_data\predict\M35.wav 52.988630533218384 %의 확률로 남자입니다.73
C:\nmb\nmb_data\predict\M36.wav 81.81235194206238 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M37.wav 99.66540336608887 %의 확률로 남자입니다.74
C:\nmb\nmb_data\predict\M38.wav 99.40950870513916 %의 확률로 남자입니다.75
C:\nmb\nmb_data\predict\M39.wav 96.99885845184326 %의 확률로 남자입니다.76
C:\nmb\nmb_data\predict\M4.wav 99.56706762313843 %의 확률로 남자입니다.77
C:\nmb\nmb_data\predict\M40.wav 99.90389347076416 %의 확률로 남자입니다.78
C:\nmb\nmb_data\predict\M41.wav 98.45073223114014 %의 확률로 남자입니다.79
C:\nmb\nmb_data\predict\M42.wav 98.74536991119385 %의 확률로 남자입니다.80
C:\nmb\nmb_data\predict\M43.wav 86.69463992118835 %의 확률로 남자입니다.81
C:\nmb\nmb_data\predict\M5.wav 94.02170181274414 %의 확률로 남자입니다.82
C:\nmb\nmb_data\predict\M5_high.wav 51.13964080810547 %의 확률로 남자입니다.83
C:\nmb\nmb_data\predict\M5_low.wav 98.41738939285278 %의 확률로 남자입니다.84
C:\nmb\nmb_data\predict\M6.wav 99.8435914516449 %의 확률로 남자입니다.85
C:\nmb\nmb_data\predict\M7_high.wav 94.04184818267822 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M7_low.wav 99.72284436225891 %의 확률로 남자입니다.86
C:\nmb\nmb_data\predict\M8.wav 91.92045331001282 %의 확률로 남자입니다.87
C:\nmb\nmb_data\predict\M9.wav 75.7852554321289 %의 확률로 남자입니다.88
time >>  0:10:06.045919

in case of original

loss :  0.2781011760234833
acc :  0.8880208134651184
C:\nmb\nmb_data\predict\F1.wav 99.41044449806213 %의 확률로 여자입니다. 1
C:\nmb\nmb_data\predict\F10.wav 56.96932673454285 %의 확률로 남자입니다. 
C:\nmb\nmb_data\predict\F11.wav 51.986223459243774 %의 확률로 여자입니다. 2
C:\nmb\nmb_data\predict\F12.wav 66.96658134460449 %의 확률로 여자입니다. 3
C:\nmb\nmb_data\predict\F13.wav 63.06905150413513 %의 확률로 여자입니다. 4
C:\nmb\nmb_data\predict\F14.wav 99.44629073143005 %의 확률로 여자입니다. 5
C:\nmb\nmb_data\predict\F15.wav 95.16807198524475 %의 확률로 여자입니다. 6
C:\nmb\nmb_data\predict\F16.wav 64.12866711616516 %의 확률로 남자입니다. 
C:\nmb\nmb_data\predict\F17.wav 96.02131247520447 %의 확률로 여자입니다. 7
C:\nmb\nmb_data\predict\F18.wav 70.99584937095642 %의 확률로 여자입니다. 8 
C:\nmb\nmb_data\predict\F19.wav 97.35345244407654 %의 확률로 여자입니다. 9
C:\nmb\nmb_data\predict\F1_high.wav 89.86919522285461 %의 확률로 여자입니다. 10
C:\nmb\nmb_data\predict\F2.wav 57.1435809135437 %의 확률로 여자입니다. 11
C:\nmb\nmb_data\predict\F20.wav 98.31114411354065 %의 확률로 여자입니다. 12
C:\nmb\nmb_data\predict\F21.wav 99.88701939582825 %의 확률로 여자입니다. 13
C:\nmb\nmb_data\predict\F22.wav 86.23999953269958 %의 확률로 여자입니다. 14
C:\nmb\nmb_data\predict\F23.wav 99.70295429229736 %의 확률로 여자입니다. 15
C:\nmb\nmb_data\predict\F24.wav 98.58946800231934 %의 확률로 여자입니다. 16
C:\nmb\nmb_data\predict\F25.wav 88.95132541656494 %의 확률로 여자입니다. 17
C:\nmb\nmb_data\predict\F26.wav 79.1258156299591 %의 확률로 여자입니다. 18
C:\nmb\nmb_data\predict\F27.wav 89.7958517074585 %의 확률로 여자입니다. 19
C:\nmb\nmb_data\predict\F28.wav 73.62164855003357 %의 확률로 여자입니다. 20
C:\nmb\nmb_data\predict\F29.wav 82.97597169876099 %의 확률로 여자입니다. 21
C:\nmb\nmb_data\predict\F2_high.wav 66.05716943740845 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F2_low.wav 50.33271908760071 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F3.wav 97.79434204101562 %의 확률로 여자입니다. 23
C:\nmb\nmb_data\predict\F30.wav 97.763192653656 %의 확률로 여자입니다. 24
C:\nmb\nmb_data\predict\F31.wav 99.8346209526062 %의 확률로 여자입니다. 25
C:\nmb\nmb_data\predict\F32.wav 98.03983569145203 %의 확률로 여자입니다. 26
C:\nmb\nmb_data\predict\F33.wav 94.11525130271912 %의 확률로 여자입니다. 27
C:\nmb\nmb_data\predict\F34.wav 99.27116632461548 %의 확률로 여자입니다. 28 
C:\nmb\nmb_data\predict\F35.wav 70.70979475975037 %의 확률로 여자입니다.29
C:\nmb\nmb_data\predict\F36.wav 62.264811992645264 %의 확률로 여자입니다.30
C:\nmb\nmb_data\predict\F37.wav 95.42267322540283 %의 확률로 여자입니다.31
C:\nmb\nmb_data\predict\F38.wav 95.48613429069519 %의 확률로 여자입니다.32
C:\nmb\nmb_data\predict\F39.wav 86.45042181015015 %의 확률로 여자입니다.33
C:\nmb\nmb_data\predict\F3_high.wav 89.37832713127136 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F4.wav 91.18971228599548 %의 확률로 여자입니다.34
C:\nmb\nmb_data\predict\F40.wav 51.36404037475586 %의 확률로 여자입니다.35
C:\nmb\nmb_data\predict\F41.wav 70.5427885055542 %의 확률로 여자입니다.36
C:\nmb\nmb_data\predict\F42.wav 99.98838901519775 %의 확률로 여자입니다.37
C:\nmb\nmb_data\predict\F43.wav 94.607675075531 %의 확률로 여자입니다.38
C:\nmb\nmb_data\predict\F5.wav 82.4754536151886 %의 확률로 남자입니다.
C:\nmb\nmb_data\predict\F6.wav 94.05233860015869 %의 확률로 여자입니다.39
C:\nmb\nmb_data\predict\F7.wav 99.24989342689514 %의 확률로 여자입니다.40
C:\nmb\nmb_data\predict\F8.wav 94.12248134613037 %의 확률로 여자입니다.41
C:\nmb\nmb_data\predict\F9.wav 61.12723350524902 %의 확률로 여자입니다.42
C:\nmb\nmb_data\predict\M1.wav 99.09884333610535 %의 확률로 남자입니다.43
C:\nmb\nmb_data\predict\M10.wav 72.05106019973755 %의 확률로 남자입니다.44
C:\nmb\nmb_data\predict\M11.wav 99.01019334793091 %의 확률로 남자입니다.45
C:\nmb\nmb_data\predict\M12.wav 98.79841804504395 %의 확률로 남자입니다.46
C:\nmb\nmb_data\predict\M13.wav 90.11918902397156 %의 확률로 남자입니다.47
C:\nmb\nmb_data\predict\M14.wav 95.6860363483429 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M15.wav 94.45899724960327 %의 확률로 남자입니다.48
C:\nmb\nmb_data\predict\M16.wav 86.87383532524109 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M17.wav 62.594419717788696 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M18.wav 99.26064014434814 %의 확률로 남자입니다.49
C:\nmb\nmb_data\predict\M19.wav 98.12934994697571 %의 확률로 남자입니다.50
C:\nmb\nmb_data\predict\M2.wav 86.51884198188782 %의 확률로 남자입니다.51
C:\nmb\nmb_data\predict\M20.wav 95.5868661403656 %의 확률로 남자입니다.52
C:\nmb\nmb_data\predict\M21.wav 87.8355085849762 %의 확률로 남자입니다.53
C:\nmb\nmb_data\predict\M22.wav 98.05887341499329 %의 확률로 남자입니다.54
C:\nmb\nmb_data\predict\M23.wav 84.12507176399231 %의 확률로 남자입니다.55
C:\nmb\nmb_data\predict\M24.wav 70.59261202812195 %의 확률로 남자입니다.56
C:\nmb\nmb_data\predict\M25.wav 98.77068400382996 %의 확률로 남자입니다.57
C:\nmb\nmb_data\predict\M26.wav 98.7497091293335 %의 확률로 남자입니다.58
C:\nmb\nmb_data\predict\M27.wav 90.91343283653259 %의 확률로 남자입니다.59
C:\nmb\nmb_data\predict\M28.wav 98.43106269836426 %의 확률로 남자입니다.60
C:\nmb\nmb_data\predict\M29.wav 68.57433319091797 %의 확률로 남자입니다.61
C:\nmb\nmb_data\predict\M2_high.wav 70.26362419128418 %의 확률로 남자입니다.62
C:\nmb\nmb_data\predict\M2_low.wav 99.76993203163147 %의 확률로 남자입니다.63
C:\nmb\nmb_data\predict\M3.wav 99.97875094413757 %의 확률로 남자입니다.64
C:\nmb\nmb_data\predict\M30.wav 77.57525444030762 %의 확률로 남자입니다.65
C:\nmb\nmb_data\predict\M31.wav 54.6538770198822 %의 확률로 여자입니다.
C:\nmb\nmb_data\predict\M32.wav 95.17754316329956 %의 확률로 남자입니다.66
C:\nmb\nmb_data\predict\M33.wav 85.13792157173157 %의 확률로 남자입니다.67
C:\nmb\nmb_data\predict\M34.wav 84.6308171749115 %의 확률로 남자입니다.68
C:\nmb\nmb_data\predict\M35.wav 81.02140426635742 %의 확률로 남자입니다.69
C:\nmb\nmb_data\predict\M36.wav 69.35502290725708 %의 확률로 남자입니다.70
C:\nmb\nmb_data\predict\M37.wav 99.49694275856018 %의 확률로 남자입니다.71
C:\nmb\nmb_data\predict\M38.wav 99.66487288475037 %의 확률로 남자입니다.72
C:\nmb\nmb_data\predict\M39.wav 94.93058919906616 %의 확률로 남자입니다.73
C:\nmb\nmb_data\predict\M4.wav 99.74270462989807 %의 확률로 남자입니다.74
C:\nmb\nmb_data\predict\M40.wav 99.91161227226257 %의 확률로 남자입니다.75
C:\nmb\nmb_data\predict\M41.wav 99.67544674873352 %의 확률로 남자입니다.76
C:\nmb\nmb_data\predict\M42.wav 95.05338072776794 %의 확률로 남자입니다.77
C:\nmb\nmb_data\predict\M43.wav 98.74948859214783 %의 확률로 남자입니다.78
C:\nmb\nmb_data\predict\M5.wav 99.24378991127014 %의 확률로 남자입니다.79
C:\nmb\nmb_data\predict\M5_high.wav 78.03582549095154 %의 확률로 남자입니다.80
C:\nmb\nmb_data\predict\M5_low.wav 98.69369864463806 %의 확률로 남자입니다.81
C:\nmb\nmb_data\predict\M6.wav 99.68565702438354 %의 확률로 남자입니다.82
C:\nmb\nmb_data\predict\M7_high.wav 71.37112021446228 %의 확률로 남자입니다.83
C:\nmb\nmb_data\predict\M7_low.wav 99.89068508148193 %의 확률로 남자입니다.84
C:\nmb\nmb_data\predict\M8.wav 94.33836340904236 %의 확률로 남자입니다.85
C:\nmb\nmb_data\predict\M9.wav 70.78195810317993 %의 확률로 남자입니다.86
time >>  0:10:56.848810

'''