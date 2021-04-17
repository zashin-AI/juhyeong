import librosa
from pydub import AudioSegment
import soundfile as sf
import os
from voice_handling import import_test, voice_sum

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