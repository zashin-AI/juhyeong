from flask import Flask, request, render_template, send_file
from scipy import misc

import joblib
import numpy as np
import librosa
import speech_recognition as sr


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
            n_fft = 512, hop_length=128
        )
        y_mel = librosa.amplitude_to_db(y_mel, ref = np.max)
        y_mel = y_mel.reshape(1, y_mel.shape[0] * y_mel.shape[1])

        prediction = model.predict(y_mel)
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


if __name__ == '__main__':
    model = joblib.load('c:/data/modelcheckpoint/project_xgb_default.data')
    app.run(debug=True)