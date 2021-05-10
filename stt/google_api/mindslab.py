from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence      # 기존 split_on_silence 는 copy 해두고, https://github.com/jiaaro/pydub/blob/master/pydub/silence.py 이 사람이 만든  split_on_silence 복사/수정
import speech_recognition as sr #pip install SpeechRecognition
from hanspell import spell_checker      # https://github.com/ssut/py-hanspell 여기있는 파일 다운받아서 이 함수를 사용할 폴더에 넣어야 함
import librosa.display
import librosa
from datetime import datetime


r = sr.Recognizer()


origin_file = librosa.util.find_files('C:\\nmb\\nmb_data\\STT\\mindslab/', ext=['wav'])
# volume_file = librosa.util.find_files('C:\\nmb\\nmb_data\\STT\\mindslab\\', ext=['wav'])

# file_list = [origin_file, volume_file]
file_list = [origin_file]

start = datetime.now()

# pairs = [[], [], []]
pairs = [[]]

a = 0 
for k in file_list:
    b = pairs[a]
   
    for j, path in enumerate(k) :
        sound_file = AudioSegment.from_wav(path)
        dbfs = sound_file.dBFS
        thresh = int(dbfs)
        if dbfs < thresh :
            thresh = thresh - 1
        audio_chunks = split_on_silence(sound_file,
            min_silence_len= 300,
            silence_thresh= dbfs - 16,
            keep_silence= 100
        )
        full_txt = []
        for i, chunk in enumerate(audio_chunks):    
            out_file = "C:\\nmb\\nmb_data\\chunk\\test\\"+ str(j) + f"chunk{i}.wav"
            chunk.export(out_file, format="wav")
            aaa = sr.AudioFile(out_file)
            with aaa as source :
                audio = r.record(aaa)
            try:
                txt = r.recognize_google(audio, language="ko-KR")
                spelled_sent = spell_checker.check(txt)
                checked_sent = spelled_sent.checked
                full_txt.append(str(checked_sent)) # 하나로 합칠 경우 사용
            except : # 너무 짧은 음성은 pass 됨 
                pass   
            b.append(checked_sent)
        print("파일 이름 : ",path[path.rfind('\\') + 1:])
        for checked_sent in full_txt:
            print(checked_sent)
    a += 1
    print('길이 !!!!!!!!!!!!!!!!!!!!  : ',len(b))


new = ''
for i in range(len(pairs[0])):
    new += pairs[0][0] + '\n\n'

with open('C:\\nmb\\nmb_data\\STT\\' + str(pairs[0][0]) + '.txt', 'wt') as f: f.writelines(new)        

end = datetime.now()

print('실행 시간 : ', end - start)

# mindslab
# 3번 파일 