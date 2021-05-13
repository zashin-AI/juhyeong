import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from pydub import AudioSegment


def speed_change(sound, speed=1.0):
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
      })

    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)


def fitch_change(loaddir, n_steps):
    files = librosa.util.find_files(loaddir, ext=['wav'])
    files = np.asarray(files)

    for file in files:
        data, rate = librosa.load(file, sr=22050)
        data = librosa.effects.pitch_shift(
        data, rate, n_steps = n_steps
    )

    return file

# y, sr = librosa.load('c:/nmb/nmb_data/korea_multi_t12_0.wav')
# y_ori = AudioSegment.from_file('c:/nmb/nmb_data/korea_multi_t12_0.wav')

# y_up = librosa.effects.pitch_shift(
#     y, sr, 5
# )
# y_down = librosa.effects.pitch_shift(
#     y, sr, -2
# )

# sf.write('c:/nmb/nmb_data/korea_multi_up.wav', y_up, sr)
# sf.write('c:/nmb/nmb_data/korea_multi_down.wav', y_down, sr)

# y_fast = speed_change(y_ori, speed = 1.3)
# y_slow = speed_change(y_ori, speed = 0.7)

# y_fast.export('c:/nmb/nmb_data/korea_multi_fast.wav', format = 'wav')
# y_slow.export('c:/nmb/nmb_data/korea_multi_slow.wav', format = 'wav')

# y_o, sr_o = librosa.load('c:/nmb/nmb_data/korea_multi_t12_0.wav')
y_f, sr_f = librosa.load('c:/nmb/nmb_data/korea_multi_fast.wav')
y_s, sr_m = librosa.load('c:/nmb/nmb_data/korea_multi_slow.wav')
# y_u, sr_u = librosa.load('c:/nmb/nmb_data/korea_multi_up.wav')
# y_d, sr_d = librosa.load('c:/nmb/nmb_data/korea_multi_down.wav')

fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

librosa.display.waveplot(y_f, sr = 22050, ax = ax1)
librosa.display.waveplot(y_s, sr = 22050, ax = ax2)

fig.tight_layout()
plt.show()