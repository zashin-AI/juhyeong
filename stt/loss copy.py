import numpy as np
import re

def custom_acc_function(txt_list):
    acc = 0
    label_length = txt_list[0]
    test_length = txt_list[1]                                   # 라벨 문자열 변수 저장
    
    def remove_string(txt):
        text = re.sub('[~.\'\",![\](\)<\>\/?\n\t ]', '', txt)
        return text

    label_length = remove_string(label_length)
    test_length = remove_string(test_length)

    if test_length[0] == ' ':
        test_length = test_length[1:]

    if len(label_length) == len(test_length):                   # 라벨과 STT 의 길이 비교 (서로 같은 경우)
        for j in range(len(label_length)):                      # 라벨 문자열의 길이 기준으로 문자 매치
            try:
                if label_length[j] == test_length[j]:           # 라벨과 STT 문자열의 j 번째끼리 비교
                    acc += 1                                    # 일치하는 경우 acc 값에 1 을 추가한다
                else:
                    acc += 0                                    # 서로 다른 경우 acc 변동 없음
            except:
                pass
    elif len(label_length) > len(test_length):
        for j in range(len(label_length)):
            try:
                if label_length[j] == test_length[j]:
                    acc += 1
                else:
                    acc += 0
                    test_length = test_length[:j] + '○' + test_length[j:]
                    # print(test_length)
                    if len(label_length) == len(test_length):
                        return custom_acc_function([label_length, test_length])
            except:
                pass
    else:
        for j in range(len(label_length)):
            try:
                if label_length[j] == test_length[j]:
                    acc += 1
                else:
                    acc += 0
                    label_length = label_length[:j] + '○' + label_length[j:]
                    if len(label_length) == len(test_length):
                        return custom_acc_function([label_length, test_length])
            except:
                pass
    acc = [np.sum(acc), np.sum(acc)/len(label_length)]

    print(f'{len(label_length)} 글자 중 맞춘 글자의 갯수 : {acc[0]}')
    print(f'정답률 : {acc[1] * 100:.3f} %')
    return acc



if __name__ == '__main__':
    label = open('c:/nmb/nmb_data/korea_multi_label.txt', 'r', encoding='utf-8')
    test = open('c:/nmb/nmb_data/korea_multi_original.txt', 'r', encoding='utf-8')
    answer = open('c:/nmb/nmb_data/answer.txt', 'r', encoding='utf-8')

    a = label.read()
    b = test.read()
    h = answer.read()

    print('stt : ', custom_acc_function([a, b]))

    c = 'abcdfgh'
    d = 'abdefgh'

    print('c, d : ')
    custom_acc_function([c, d])

    e = 'abcdef'
    f = ' abcde'

    custom_acc_function([e, f])


    # e = '[오늘은 의심스러웠지만~, 그만 문을 열어 주고 말았어요]'
    # f = '[오는 의심스러웠지만 그만 문을 열어 주고 말았어요]'

    # custom_acc_function([e, f])
    # custom_acc_function([f, e])
