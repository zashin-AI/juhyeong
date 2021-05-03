import speech_recognition as sr
import librosa
import soundfile as sf

from hanspell import spell_checker
from pydub import AudioSegment
from librosa.core import audio
from numpy import lib

r = sr.Recognizer()

f = open('c:/nmb/nmb_data/민선이바보.txt', 'w')

stt_list = list()
for i in range(1, 11):
    f.write('[' + str(i) + ']' + '\n')
    audio = sr.AudioFile(
        'c:/nmb/nmb_data/STT/STT_F_pred/F_wav/F_1_file/' + str(i) + '.wav'
    )

    with audio as audio_file:
        file = r.record(audio_file)
        stt = r.recognize_google(file, language='ko-KR')

    spell_check = spell_checker.check(stt)
    stt_spell = spell_check.checked
    
    print(stt_spell)
    f.write('stt' + '\t' + ' : ' + stt + '\n' + 'hanspell   : ' + stt_spell + '\n\n')