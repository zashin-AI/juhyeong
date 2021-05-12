import numpy as np
import re

def custom_acc_function(txt_list):
    acc = 0
    label_length = txt_list[0]
    test_length = txt_list[1]                                   # 라벨 문자열 변수 저장
    
    # def remove_string(txt):
    #     text = re.sub('[~.\'\",![\](\)<\>\/?\n\t ]', '', txt)
    #     return text

    # label_length = remove_string(label_length)
    # test_length = remove_string(test_length)

    if test_length[0] == ' ':
        test_length = test_length[1:]

    label_length = label_length.split()
    test_length = test_length.split()

    print(label_length)
    print(test_length)

    acc = 0

    for i in test_length:
        if i in label_length:
            for j in range(len(label_length)):
                if label_length[j] == test_length[j]:
                    acc += 1

    print(f'{len(label_length)} 글자 중 맞춘 글자의 갯수 : {acc}')
    print(f'정답률 : {np.sum(acc)/len(label_length) * 100:.3f} %')
    return acc



if __name__ == '__main__':
    # label = open('c:/nmb/nmb_data/korea_multi_label.txt', 'r', encoding='utf-8')
    # test = open('c:/nmb/nmb_data/korea_multi_original.txt', 'r', encoding='utf-8')
    # answer = open('c:/nmb/nmb_data/answer.txt', 'r', encoding='utf-8')

    # a = label.read()
    # b = test.read()
    # h = answer.read()

    # print('stt : ')
    # custom_acc_function([a, b])

    c = '밥을 먹었어요 '
    d = '밥을 아 먹었어'

    print('c, d : ')
    custom_acc_function([c, d])

    # e = '오나라 오나라 아주 오나'
    # f = '오라 오라 오라'

    # custom_acc_function([e, f])

    # e = 'abcdef '
    # # f = ' abcdef'
    # f = ' abcedf'
    
    # print('e, f : ')
    # custom_acc_function([e, f])

    # txt_label = open('c:/nmb/test_label.txt', 'r', encoding='utf-8')
    # txt_test = open('c:/nmb/test_test.txt', 'r', encoding='utf-8')

    # ttl = txt_label.read()
    # ttt = txt_test.read()

    # custom_acc_function([ttl, ttt])

    # e = '[오늘은 의심스러웠지만~, 그만 문을 열어 주고 말았어요]'
    # f = '[오는 의심스러웠지만 그만 문을 열어 주고 말았어요]'

    # custom_acc_function([e, f])
    # custom_acc_function([f, e])