import librosa
import librosa.display
import matplotlib.pyplot as plt

gan1, sr1 = librosa.load(
    'C:/nmb/nmb_data/checkpoints/20-04-2021_0h/synth_audio/0_batch_synth_class_0.wav'
)

gan2, sr2 = librosa.load(
    'C:/nmb/nmb_data/checkpoints/20-04-2021_0h/synth_audio/18000_batch_synth_class_1.wav'
)

fig = plt.figure(figsize = (16, 6))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

librosa.display.waveplot(gan1, sr = sr1, ax = ax1)
librosa.display.waveplot(gan2, sr = sr2, ax = ax2)

fig.tight_layout()
plt.show()