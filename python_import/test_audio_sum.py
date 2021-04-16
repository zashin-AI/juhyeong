import librosa
from pydub import AudioSegment
import soundfile as sf
import os
from voice_handling import import_test, voice_sum

import_test()
# ==== it will be great ====

# ---------------------------------------------------------------
# voice_sum: 오디오 한 wav 파일로 합쳐서 저장하기
# def voice_sum(form, pathaudio, save_dir, out_dir):
# **** example ****
# form(파일 형식): 'wav' or 'flac'
# audio_dir(여러 오디오가 있는 파일경로) = 'C:/nmb/nmb_data/F1F2F3/F3/'
# save_dir(flac일 경우 wav파일로 저장할 경로) = 'C:/nmb/nmb_data/F1F2F3/F3_to_wave/'
# out_dir(wav파일을 합쳐서 저장할 경로+파일명까지) = "C:/nmb/nmb_data/combine_test/F3_sum.wav"

# 1) wav일 때
# path_wav = 'D:/nmb_test/test_wav/'
# path_out = 'D:/nmb_test/test_sum/test_01_wav_sum.wav'
# voice_sum(form='wav', audio_dir=path_wav, save_dir=None, out_dir=path_out)
# # 잘 되는 것 확인!

# # 2) flac일 때
# audio_dir = 'D:/nmb_test/test_flac/'
# save_dir = 'D:/nmb_test/test_flac_to_wav/'
# out_dir = 'D:/nmb_test/test_sum/test_02_flac_to_wave_sum.wav'
# voice_sum(form='flac', audio_dir=path_flac, save_dir=path_save, out_dir=path_out)
# # 잘 되는 것 확인!

# for i in range(1, 33):
#     try:
#         if not(os.path.isdir('c:/nmb/nmb_data/pansori/m' + str(i))):
#             os.makedirs(os.path.join('c:/nmb/nmb_data/pansori/m' + str(i)))
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             print("Failed to create directory!!!!!")
#             raise


# for i in range(1, 33):
#     form = 'flac'   
#     audio_dir = 'c:/nmb/nmb_data/pansori_original/pansori_male/m' + str(i) + '/'
#     save_dir = 'c:/nmb/nmb_data/pansori/male_wav/m' + str(i) + '/'
#     out_dir = 'c:/nmb/nmb_data/pansori/male/pansori_male_' + str(i) + '.wav'

#     voice_sum(
#         form = form,
#         audio_dir = audio_dir,
#         save_dir = save_dir,
#         out_dir = out_dir
#     )