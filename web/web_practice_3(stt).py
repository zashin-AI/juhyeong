from flask import Flask, request, render_template, send_file
from scipy import misc
from tensorflow.keras.models import load_model
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence

import numpy as np
import librosa
import speech_recognition as sr
import tensorflow as tf
import os
import soundfile as sf
import copy
import random

import sys
sys.path.append('c:/nmb/nada/python_import/')

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


# 필요함수 정의
r = sr.Recognizer()

def normalized_sound(auido_file):
    '''
    볼륨 정규화
    '''
    audio = AudioSegment.from_wav(auido_file)
    normalizedsound = effects.normalize(audio)
    return normalizedsound

def split_slience(audio_file):
    '''
    음성 파일 묵음마다 자르기
    '''
    dbfs = audio_file.dBFS
    audio_chunks = split_on_silence(
        audio_file,
        min_silence_len=1000,
        silence_thresh=dbfs - 30,
        keep_silence=True
    )
    return audio_chunks

def STT(audio_file):
    '''
    STT 실행
    '''
    with audio_file as audio:
        file = r.record(audio)
        stt = r.recognize_google(file, language='ko-KR')
    return stt

def predict_speaker(y, sr):
    '''
    화자 구분
    '''
    mels = librosa.feature.melspectrogram(y, sr = sr, hop_length=128, n_fft=512, win_length=512)
    pred_mels = librosa.amplitude_to_db(mels, ref=np.max)
    pred_mels = pred_mels.reshape(1, pred_mels.shape[0], pred_mels.shape[1])

    y_pred = model.predict(pred_mels)
    y_pred_label = np.argmax(y_pred)
    if y_pred_label == 0:
        return '여자'
    if y_pred_label == 1:
        return '남자'

app = Flask(__name__)


# 첫 화면 (파일 업로드)
@app.route('/')
def upload_file():
    return render_template('upload.html')

# 업로드 후에 출력 되는 화면 (추론)
@app.route('/uploadFile', methods = ['POST'])
def download():
    # 파일이 업로드 되면 실시 할 과정
    if request.method == 'POST':
        f = request.files['file']
        if not f: return render_template('upload.html')

        f_path = os.path.splitext(str(f))
        f_path = os.path.split(f_path[0])

        folder_path = 'c:/nmb/nmb_data/web/chunk/'
        f_name = f_path[1] 

        normalizedsound = normalized_sound(f)
        audio_chunks = split_slience(normalizedsound)
        len_audio_chunks = len(audio_chunks)

        save_script = ''

        for i, chunk in enumerate(audio_chunks):
            speaker_stt = list()
            out_file = folder_path + '/' + str(i) + '_chunk.wav'
            chunk.export(out_file, format = 'wav')
            aaa = sr.AudioFile(out_file)

            try:
                stt_text = STT(aaa)
                speaker_stt.append(str(stt_text))

                y, sample_rate = librosa.load(out_file, sr = 22050)

                if len(y) >= 22050*5:
                    y = y[:22050*5]
                    speaker = predict_speaker(y, sample_rate)
                    speaker_stt.append(str(speaker))
                    print(speaker_stt[1], " : ", speaker_stt[0])

                else:
                    audio_copy = AudioSegment.from_wav(out_file)
                    audio_copy = copy.deepcopy(audio_copy)
                    for num in range(3):
                        audio_copy = audio_copy.append(copy.deepcopy(audio_copy), crossfade=0)
                    audio_copy.export(folder_path + '/' + str(i) + '_cunks_over_5s.wav', format = 'wav')
                    y_copy, sample_rate = librosa.load(folder_path + '/' + str(i) + '_cunks_over_5s.wav')
                    y_copy = y_copy[:22050*5]
                    speaker = predict_speaker(y_copy, sample_rate)
                    speaker_stt.append(str(speaker))
                    print(speaker_stt[1] + " : " + speaker_stt[0])

                save_script += speaker_stt[1] + " : " + speaker_stt[0] + '\n\n'
                with open('c:/nmb/nada/web/static/test.txt', 'wt', encoding='utf-8') as f: f.writelines(save_script)

            except:
                pass
        return render_template('/download.html')
    

# 파일 다운로드
@app.route('/download')
def download_file():
        pp = 'c:/nmb/nada/web/static/test.txt'
        return send_file(
            pp, as_attachment = True, mimetype='text/txt'
        )

# 추론 된 파일 읽기
@app.route('/read')
def read_text():
    f = open('C:/nmb/nada/web/static/test.txt', 'r', encoding='utf-8')
    return "</br>".join(f.readlines())    


if __name__ == '__main__':
    model = load_model('c:/data/modelcheckpoint/mobilenet_rmsprop_1.h5')
    app.run(debug=True)