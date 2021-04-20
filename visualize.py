import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

gan, sr = librosa.load(
    'c:/nmb/nmb_data/predict/F/F7.wav'
)

# gan = np.abs(librosa.stft(gan, n_fft = 512, hop_length = 128))
# gan = np.abs(np.fft.fft(gan))
gan = librosa.feature.melspectrogram(
    gan,
    sr = sr,
    n_fft = 512,
    hop_length = 128,
    win_length = 512
)
# def normalize(x, axis=0):
#     return sklearn.preprocessing.minmax_scale(x, axis=axis)

# gan = librosa.feature.mfcc(
#     gan,
#     sr = sr
# )
# gan = normalize(gan)

# fig = plt.figure(figsize = (16, 6))

# librosa.display.waveplot(gan, sr = sr)
# librosa.display.waveplot(gan2, sr = sr2, ax = ax2)
librosa.display.specshow(gan, sr = sr)

# plt.plot(gan)
# plt.plot(gan2, ax = ax2)

# .tight_layout()
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