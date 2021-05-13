
import numpy as np
import datetime 
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D, Dropout, Activation, Flatten, Add, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import all_estimators    
import warnings
warnings.filterwarnings('ignore')
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

start_now = datetime.datetime.now()

# 데이터 불러오기
x = np.load('c:/nmb/nmb_data/npy/total_data.npy')
x = x.reshape(-1, x.shape[1] * x.shape[2])
y = np.load('c:/nmb/nmb_data/npy/total_label.npy')

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
print(x_train.shape)    # (3628, 110336)
print(x_test.shape)     # (908, 110336)
print(y_train.shape)    # (3628, 110336)
print(y_test.shape)     # (908, 110336)


# 모델 구성

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms :    # 분류형 모델 전체를 반복해서 돌린다.
    # try ... except... : 예외처리 구문
    try :   # 에러가 없으면 아래 진행됨
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test, y_pred))
    except :          #에러가 발생하면
        # continue    # 정지시키지 않고 계속 진행시키겠다.
        print(name, "은 없는 모델") # 예외처리한 모델 이름을 출력 



allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms :    
    try :   
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test, y_pred))
    except :          
        print(name, "은 없는 모델") 

end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >

'''
AdaBoostClassifier 의 정답률 :  0.8964757709251101
BaggingClassifier 의 정답률 :  0.8865638766519823
BernoulliNB 의 정답률 :  0.5506607929515418
CalibratedClassifierCV 의 정답률 :  0.9251101321585903
CategoricalNB 은 없는 모델
ClassifierChain 은 없는 모델
ComplementNB 은 없는 모델
DecisionTreeClassifier 의 정답률 :  0.8381057268722467
DummyClassifier 의 정답률 :  0.5506607929515418
ExtraTreeClassifier 의 정답률 :  0.775330396475771
ExtraTreesClassifier 의 정답률 :  0.9118942731277533
GaussianNB 의 정답률 :  0.5099118942731278
GaussianProcessClassifier 의 정답률 :  0.5506607929515418
GradientBoostingClassifier 의 정답률 :  0.9273127753303965
HistGradientBoostingClassifier 의 정답률 :  0.9229074889867841
KNeighborsClassifier 의 정답률 :  0.8370044052863436
LabelPropagation 의 정답률 :  0.5506607929515418
LabelSpreading 의 정답률 :  0.5506607929515418
LinearDiscriminantAnalysis 의 정답률 :  0.9162995594713657
LinearSVC 의 정답률 :  0.9185022026431718
LogisticRegression 의 정답률 :  0.9129955947136564
LogisticRegressionCV 의 정답률 :  0.9185022026431718
MLPClassifier 의 정답률 :  0.44933920704845814
MultiOutputClassifier 은 없는 모델
MultinomialNB 은 없는 모델
NearestCentroid 의 정답률 :  0.802863436123348
NuSVC 의 정답률 :  0.9151982378854625
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  0.9251101321585903
Perceptron 의 정답률 :  0.7775330396475771
QuadraticDiscriminantAnalysis 의 정답률 :  0.44933920704845814
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 :  0.9052863436123348
RidgeClassifier 의 정답률 :  0.8931718061674009
RidgeClassifierCV 의 정답률 :  0.8931718061674009
SGDClassifier 의 정답률 :  0.9107929515418502
SVC 의 정답률 :  0.9427312775330396
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''