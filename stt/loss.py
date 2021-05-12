import numpy as np

def custom_acc_function(txt_list):
    acc = 0
    for i in range(len(txt_list)//2):
        label_length = txt_list[0]
        test_length = txt_list[1]                                   # 라벨 문자열 변수 저장
        if len(label_length) == len(txt_list[i]):                   # 라벨과 STT 의 길이 비교 (서로 같은 경우)
            for j in range(len(label_length)):                      # 라벨 문자열의 길이 기준으로 문자 매치
                try:
                    if label_length[j] == test_length[j]:           # 라벨과 STT 문자열의 j 번째끼리 비교
                        acc += 1                                    # 일치하는 경우 acc 값에 1 을 추가한다
                    else:
                        acc += 0                                    # 서로 다른 경우 acc 변동 없음
                except:
                    pass
        elif len(label_length) > len(txt_list[i]):                  # 라벨과 STT 의 길이 비교 (라벨 > STT)
            label_length = label_length[:len(txt_list[i])]          # 라벨의 길이가 더 기므로 STT 의 길이까지만 우선 비교
            for k in range(len(label_length)):
                try:
                    if label_length[k] == test_length[k]:
                        acc += 1
                    else:
                        acc += 0
                except:
                    pass
            acc += len(label_length[len(txt_list[i]):])
        else:                                                       # 라벨과 STT 의 길이 비교 (라벨 < STT)
            label_length = label_length[len(txt_list[i]):]          # 라벨의 길이가 더 짧으므로 라벨의 길이만큼만 STT 와 우선 비교
            for h in range(len(label_length)):
                try:
                    if label_length[h] == test_length[h]:
                        acc += 1
                    else:
                        acc += 0
                except:
                    pass
            acc += len(label_length[:len(txt_list[i])])
        acc = [np.sum(acc), np.sum(acc)/len(label_length)]          # 전체 글자 중에 맞춘 갯수, 정답률

    print('{} 글자 중 맞춘 글자의 갯수 : {}'.format(len(txt_list[0]), acc[0]))
    print('정답률 : {}%'.format(acc[1] * 100))
    return acc

label = open('c:/nmb/nmb_data/korea_multi_label.txt', 'r', encoding='utf-8')
test = open('c:/nmb/nmb_data/korea_multi_original.txt', 'r', encoding='utf-8')

a = label.read()
b = test.read()

print(len(a))
print(len(b))
print(custom_acc_function([a, b]))

c = 'abcde'
d = 'aadce'

print(custom_acc_function([c, d]))