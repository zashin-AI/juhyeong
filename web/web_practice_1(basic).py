from flask import Flask, request, render_template, send_file
from scipy import misc
from tensorflow.keras.models import load_model

import joblib
import numpy as np
import librosa
import speech_recognition as sr
import tensorflow as tf

# gpu failed init~~ 에 관한 에러 해결
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


app = Flask(__name__)

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploadFile', methods = ['POST'])
def make_predict():
    if request.method == 'POST':
        f = request.files['file']
        if not f: return render_template('upload.html')

        y, sr = librosa.load(f)
        y_mel = librosa.feature.melspectrogram(
            y, sr = sr,
            n_fft = 512, hop_length=128, win_length=512
        )
        y_mel = librosa.amplitude_to_db(y_mel, ref = np.max)
        y_mel = y_mel.reshape(1, y_mel.shape[0], y_mel.shape[1])

        prediction = model.predict(y_mel)
        prediction = np.argmax(prediction)
        
        if prediction == 0:
            with open('c:/nmb/nmb_data/web/test.txt', 'w') as p:
                p.write('여자다 이 자식아')
        elif prediction == 1:
            with open('c:/nmb/nmb_data/web/test.txt', 'w') as p:
                p.write('남자다 이 자식아')

        pp = 'c:/nmb/nmb_data/web/test.txt'
        print(prediction)

        return send_file(
            pp, as_attachment = True, mimetype='text/txt'
        )

@app.route('/read')
def read_text():
    f = open('C:/Users/ai/Downloads/test.txt', 'r')#, encoding='utf-8')
    return "</br>".join(f.readlines())    


if __name__ == '__main__':
    model = load_model('c:/data/modelcheckpoint/mobilenet_rmsprop_1.h5')
    app.run(debug=True)