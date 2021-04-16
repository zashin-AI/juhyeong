from feature_handling import load_data_mfcc

# filepath = 파일 불러올 경로
# filename = 파일 확장자명 (wav, flac...)
# labels = 라벨링 (여자 0, 남자 1)

filepath = 'C:/nmb/nmb_data/ForM/M/' # 파일 경로
filename = 'flac' # 파일 확장자
labels = 1 # 라벨값 지정

dataset, label = load_data_mfcc(filepath, filename, labels)

print(dataset)
print(label)
print(type(dataset))
print(type(label))
