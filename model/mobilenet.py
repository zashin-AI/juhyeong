from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split

import librosa
import numpy as np
import datetime

str_time = datetime.datetime.now()

# data
x = np.load('c:/nmb/nmb_data/npy/total_data.npy')
y = np.load('c:/nmb/nmb_data/npy/total_label.npy')

print(x.shape) # (4536, 128, 862)
print(y.shape) # (4536)

x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state=42
)

# modeling
model = MobileNet(
    include_top=True,
    input_shape=(128, 862, 1),
    classes=2,
    pooling=None,
    weights=None
)

model.save('c:/data/h5/mobilenet.h5')

# compile, train
op = RMSprop(learning_rate=1e-3)
batch_size = 8

es = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

lr = ReduceLROnPlateau(
    factor = 0.5,
    verbose = 1
)

path = 'c:/data/h5/mobilenet_mc.h5'
mc = ModelCheckpoint(
    path, save_best_only=True, verbose=1
)

model.compile(
    optimizer=op,
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

history = model.fit(
    x_train, y_train,
    epochs=1000, batch_size=batch_size, validation_split=0.2, callbacks=[es, lr, mc]
)

model = load_model('c:/data/h5/mobilenet.h5')
results = model.evaluate(x_test, y_test, batch_size=batch_size)

print(f'loss : {results.shape[0]}')
print(f'acc : {results.shape[1]}')