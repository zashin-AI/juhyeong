import librosa
import librosa.display
import matplotlib.pyplot as plt

gan, sr1 = librosa.load(
    'c:/nmb/nmb_data/audio_data/gan/800_batch_synth_class_0.wav'
)

gan_denoise, sr2 = librosa.load(
    'c:/nmb/nmb_data/gan_denoise/_noise/800_batch_synth_class_0_noise.wav'
)

fig = plt.figure(figsize = (16, 6))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

librosa.display.waveplot(gan, sr = sr1, ax = ax1)
librosa.display.waveplot(gan_denoise, sr = sr2, ax = ax2)

fig.tight_layout()
plt.show()