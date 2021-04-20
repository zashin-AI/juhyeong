import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

########################## audio visualization ############################
gan, sr1 = librosa.load(
    'c:/nmb/nmb_data/38000_batch_synth_class_0.wav'
)
# gan = gan * 100

gan2, sr2 = librosa.load(
    'c:/nmb/nmb_data/39000_batch_synth_class_0.wav'
)

# gan = np.abs(librosa.stft(gan, n_fft = 512, hop_length = 128))
# gan = np.abs(np.fft.fft(gan))

fig = plt.figure(figsize = (16, 6))

librosa.display.waveplot(gan, sr = sr1)
# librosa.display.waveplot(gan2, sr = sr2, ax = ax2)
# plt.plot(gan)
# plt.plot(gan2, ax = ax2)

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