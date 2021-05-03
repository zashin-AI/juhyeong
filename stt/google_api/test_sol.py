import speech_recognition as sr
import librosa
import soundfile as sf

from librosa.core import audio
from numpy import lib

r = sr.Recognizer()

audio = sr.AudioFile(
    'c:/nmb/nmb_data/'
)