import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

<<<<<<< HEAD
gan1, sr1 = librosa.load(
    'C:/nmb/nmb_data/checkpoints/20-04-2021_0h/synth_audio/0_batch_synth_class_0.wav'
=======
########################## audio visualization ############################
gan, sr1 = librosa.load(
    'c:/nmb/nmb_data/38000_batch_synth_class_0.wav'
>>>>>>> a0fd4aad21846877bf9e8760cd0e6777351255d4
)
# gan = gan * 100

gan2, sr2 = librosa.load(
<<<<<<< HEAD
    'C:/nmb/nmb_data/checkpoints/20-04-2021_0h/synth_audio/18000_batch_synth_class_1.wav'
=======
    'c:/nmb/nmb_data/39000_batch_synth_class_0.wav'
>>>>>>> a0fd4aad21846877bf9e8760cd0e6777351255d4
)

# gan = np.abs(librosa.stft(gan, n_fft = 512, hop_length = 128))
# gan = np.abs(np.fft.fft(gan))
# gan = librosa.feature.melspectrogram(
#     gan,
#     sr = sr1,
#     n_fft = 512,
#     hop_length = 128,
#     win_length = 512
# )
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

gan = librosa.feature.mfcc(
    gan,
    sr = sr1
)
gan = normalize(gan)

fig = plt.figure(figsize = (16, 6))

<<<<<<< HEAD
librosa.display.waveplot(gan1, sr = sr1, ax = ax1)
librosa.display.waveplot(gan2, sr = sr2, ax = ax2)
=======
# librosa.display.waveplot(gan, sr = sr1)
# librosa.display.waveplot(gan2, sr = sr2, ax = ax2)
librosa.display.specshow(gan, sr = sr1)

# plt.plot(gan)
# plt.plot(gan2, ax = ax2)


>>>>>>> a0fd4aad21846877bf9e8760cd0e6777351255d4

fig.tight_layout()
plt.show()

######################### csv visualization ############################

# df = pd.read_csv(
#     'c:/nmb/nmb_data/loss_2.csv'
# )

# print(df.info())
# print(df)

# fig = plt.figure(figsize = (16, 6))
# df_vis_d = df['d_loss'].plot()
# df_vis_g = df['g_loss'].plot()
# ax1 = df_vis_d.get_figure()
# ax2 = df_vis_g.get_figure()

# plt.legend(loc = 'best')
# plt.show()