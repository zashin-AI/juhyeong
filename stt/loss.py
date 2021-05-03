import librosa
import librosa.display
import numpy as np
import speech_recognition as sr

from pydub import AudioSegment

a = 'abcd ' # label
b = 'acd b' # train_data : 1
c = 'b'     # train_data : 2
d = 'ab'

txt_list = [a, b, c, d]
print(txt_list)
print(txt_list[0][0])
print(txt_list[0][1])
print(txt_list[1][0])
print(txt_list[0][-1])

# all_length = list()
loss = 0
num = 0
for i in range(len(txt_list)):
    label_length = txt_list[0]
    # length = len(txt_list[i])
    # all_length.append(length)
    # all_length.sort()
    if len(label_length) == len(txt_list[i]):
        for j in range(len(label_length)):
            try:
                if label_length[j] == txt_list[num+1][j]:
                    loss += 1
                else:
                    loss += 0
            except:
                pass
            num+=1
    elif len(label_length) > len(txt_list[i]):
        label_length = label_length[:len(txt_list[i])]
        for k in range(len(label_length)):
            try:
                if label_length[j] == txt_list[num+1][j]:
                    loss += 1
                else:
                    loss += 0
            except:
                pass
            num += 1
        loss += len(label_length[len(txt_list[i]):])
    else:
        label_length = label_length[len(txt_list[i]):]
        for h in range(len(label_length)):
            try:
                if label_length[j] == txt_list[num+1][j]:
                    loss += 1
                else:
                    loss += 0
            except:
                pass
            num += 1
        loss += len(label_length[:len(txt_list[i])])


        

print(sorted(txt_list))
# print(all_length)
print(loss)
