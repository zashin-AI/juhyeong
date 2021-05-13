import speech_recognition as sr
import librosa
import soundfile as sf
import numpy as np

from hanspell import spell_checker

import sys
sys.path.append('c:/nmb/nada/stt/')
from loss import custom_acc_function

label_txt = open('c:/nmb/nmb_data/korea_multi_t12.txt', 'r', encoding='UTF-8')
label = label_txt.read()

r = sr.Recognizer()

# audio = sr.AudioFile(
#     'c:/nmb/nmb_data/chunk/test_speed_up/korea_multi_fast/korea_multi_fast_0.wav'
# )
# audio = sr.AudioFile(
#     'c:/nmb/nmb_data/chunk/test_speed_up/korea_multi_slow/korea_multi_slow_0.wav'
# )
# audio = sr.AudioFile(
#     'c:/nmb/nmb_data/chunk/test_speed_up/korea_multi_up/korea_multi_up_0.wav'
# )
# audio = sr.AudioFile(
#     'c:/nmb/nmb_data/chunk/test_speed_up/korea_multi_down/korea_multi_down_0.wav'
# )

# with audio as audio_file:
#     file = r.record(audio_file)
#     stt = r.recognize_google(file, language='ko-KR')

# txt.write('[' + stt + ' ]\n')
# txt.close()

# for i in range(28):
#     audio = sr.AudioFile(
#         'c:/nmb/nmb_data/chunk/test/korea_multi_t12_0/korea_multi_t12_0_{}.wav'.format(i)
#     )
#     print(str(i))
#     with audio as audio_file:
#         file = r.record(audio_file)
#         stt = r.recognize_google(file, language='ko-KR')

# spell_check = spell_checker.check(stt)
# stt_spell = spell_check.checked
# stt_list.append(stt_spell)

# print(stt_spell)
# print(label)

# print(custom_acc_function([label, stt_spell]))