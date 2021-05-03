import librosa
import soundfile as sf
import os

from pydub import AudioSegment

def wav_sum(form, audio_dir, out_dir):
    if form == 'wav':
        infiles = librosa.util.find_files(audio_dir)
        wavs = [AudioSegment.from_wav(wav) for wav in infiles]
        combined = wavs[0]
        for wav in wavs[1:]:
            combined = combined.append(wav)
        combined.export(out_dir, format = 'wav')

wav_list = ['7ISc66qF7IiZ','7J207IOB7J2A','7J207ISg7ZWY','7KCV6rSR7ZmU',\
    '105','121','126','132','137','147']

for i in wav_list:
    wav_sum(
        'wav',
        'c:/nmb/nmb_data/STT/STT_F_pred/F_wav/sum/' + str(i) + '/',
        'c:/nmb/nmb_data/STT/' + str(i) + '.wav'
    )