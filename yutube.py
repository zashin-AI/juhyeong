from pytube import YouTube
import glob
import os.path
from pydub import AudioSegment

# 먼저 실행 1번
# 유튜브 전용 인스턴스 생성
par = 'https://www.youtube.com/watch?v=SXkrpBBRPbI'
yt = YouTube(par)
yt.streams.filter()

yt.streams.filter().first().download()
print('success')

# 그 다음 실행 2번
import moviepy.editor as mp

clip = mp.VideoFileClip("[인터뷰 풀영상] 정우성 본분은 영화배우 영화 안에 있을 때 가장 빛난다고 생각해.mp4")
clip.audio.write_audiofile("audio.wav")

def voice_split_term(origin_dir, out_dir, start, end):
    audio = AudioSegment.from_file(origin_dir)
    _, w_id = os.path.split(origin_dir)
    w_id = w_id[:-4]
    start = start
    end = end
    counter = 0
    print(start, end)
    chunk = audio[start:end]
    filename = out_dir + w_id + '.wav'
    chunk.export(filename, format='wav')
    print('==== wav split done ====')

start1 = 138000

voice_split_term( # 2:18 = 138
    origin_dir='c:/nmb/nada/audio.wav',
    out_dir = 'c:/nmb/nada/audio_term3',
    start = start1,
    end = start1 + 5000
)