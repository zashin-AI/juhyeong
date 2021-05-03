import speech_recognition as sr
import librosa
import soundfile as sf
import numpy as np

from hanspell import spell_checker
from pydub import AudioSegment
from librosa.core import audio
from numpy import lib

import sys
sys.path.append('c:/nmb/nada/stt/')
from loss import custom_acc_function 

speed = 0.8

def speed_change(sound, speed=1.0):
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
      })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

a = np.arange(0.7, 1.3, 0.1)
r = sr.Recognizer()

read = open('c:/nmb/nmb_data/STT/STT_F_pred/script.txt', 'r', encoding='UTF-8')

txt_list = list()

# label 파일 불러오기
while True:
    line = read.readline()
    txt_list.append(line.rstrip('\n')) # 개행문자 제거
    if not line : break
    print(line)

print(txt_list)

f = open('c:/nmb/nmb_data/stt_speed_change.txt', 'w')
for i in range(1, 11):
    stt_list = list()
    f.write('[' + str(i) + ']' + '\n')
    for j in a:
        f.write('[' + str(np.round(j, 2)) + ']' + '\n')
        sound = AudioSegment.from_file(
            'c:/nmb/nmb_data/STT/STT_F_pred/F_wav/F_1_file/' + str(i) + '.wav'
        )

        change_speed = speed_change(sound, j)
        change_speed.export(
            'c:/nmb/nmb_data/STT/STT_F_pred/F_wav/' + str(i) + '.wav', format = 'wav'
        )

        audio = sr.AudioFile(
            'c:/nmb/nmb_data/STT/STT_F_pred/F_wav/' + str(i) + '.wav'
        )

        with audio as audio_file:
            file = r.record(audio_file)
            stt = r.recognize_google(file, language='ko-KR')

        spell_check = spell_checker.check(stt)
        stt_spell = spell_check.checked
        stt_list.append(stt_spell)

        print(stt_spell)
        print(stt_list)
        # custom_acc_function(txt_list)

        f.write('stt' + '\t' + ' : ' + stt + '\n' + 'hanspell   : ' + stt_spell + '\n\n')
f.close()
read.close()