import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn


gan1, sr = librosa.load(
    'c:/nmb/nmb_data/checkpoints2/5000.wav'
)

gan2, sr = librosa.load(
    'c:/nmb/nmb_data/checkpoints2/10000.wav'
)

gan3, sr = librosa.load(
    'c:/nmb/nmb_data/checkpoints2/20000.wav'
)

gan4, sr = librosa.load(
    'c:/nmb/nmb_data/checkpoints2/30000.wav'
)

gan1 = np.fft.fft(gan1)
gan1 = abs(gan1)
gan1_r = np.fft.fftfreq(len(gan1), d = 1.0)

gan2 = np.fft.fft(gan2)
gan2 = abs(gan2)
gan2_r = np.fft.fftfreq(len(gan2), d = 1.0)

gan3 = np.fft.fft(gan3)
gan3 = abs(gan3)
gan3_r = np.fft.fftfreq(len(gan3), d = 1.0)

gan4 = np.fft.fft(gan4)
gan4 = abs(gan4)
gan4_r = np.fft.fftfreq(len(gan4), d = 1.0)

# fig = plt.figure(figsize = (16, 6))

# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
# ax4 = fig.add_subplot(2, 2, 4)

# ax1.set(title = '5000')
# ax2.set(title = '10000')
# ax3.set(title = '20000')
# ax4.set(title = '30000')

# librosa.display.waveplot(gan1, sr, ax = ax1)
# librosa.display.waveplot(gan2, sr, ax = ax2)
# librosa.display.waveplot(gan3, sr, ax = ax3)
# librosa.display.waveplot(gan4, sr, ax = ax4)

# ax1.plot(gan1_r, gan1)
# ax2.plot(gan2_r, gan2)
# ax3.plot(gan3_r, gan3)
# ax4.plot(gan4_r, gan4)

plt.plot(gan4_r, gan4)
plt.xlim(0, 0.5)
plt.ylim(0, 100)

# fig.tight_layout()
# plt.xlim(0, 5)
# plt.ylim(0, 100)
plt.show()

######################### csv visualization ############################

'''
df = pd.read_csv(
    'c:/nmb/nmb_data/loss_2.csv'
)

print(df.info())
print(df)

fig = plt.figure(figsize = (16, 6))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

df_vis_d = df['d_loss'].plot()
df_vis_g = df['g_loss'].plot()
ax1 = df_vis_d.get_figure()
ax2 = df_vis_g.get_figure()

plt.legend(loc = 'best')
plt.show()
'''

# df = pd.read_csv(
#     'c:/nmb/nmb_data/loss_2.csv'
# )

# fig = plt.figure(figsize=(16, 6))

# # plt.title('generator_loss')

# df1 = df.loc[:5001, :]
# df2 = df.loc[:10001, :]
# df3 = df.loc[:20001, :]
# df4 = df.loc[:30001, :]

# df1 = df1['d_loss']
# df2 = df2['d_loss']
# df3 = df3['d_loss']
# df4 = df4['d_loss']

# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
# ax4 = fig.add_subplot(2, 2, 4)

# ax1.plot(df1, color = 'orange')
# ax2.plot(df2, color = 'orange')
# ax3.plot(df3, color = 'orange')
# ax4.plot(df4, color = 'orange')

# ax1.set_title('5000')
# ax2.set_title('10000')
# ax3.set_title('20000')
# ax4.set_title('30000')

# fig.tight_layout()
# plt.legend(loc = 'best')
# plt.show()


'''
d_loss = df['d_loss']
g_loss = df['g_loss']

ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

ax1.plot(d_loss, color = 'b')
plt.title('d_loss')
ax2.plot(g_loss, color = 'orange')
plt.title('g_loss')

fig.tight_layout()
plt.show()
'''