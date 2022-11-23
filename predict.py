from tensorflow import keras
import firebase_admin
from firebase_admin import credentials ,firestore
import keyinfo as key
import numpy as np
import pipeline as pipe
from sklearn.preprocessing import MinMaxScaler

cred = credentials.Certificate("key/hackathon-11d72-firebase-adminsdk-7hnwq-45f6fa260b.json")
firebase_admin.initialize_app(cred,{
    'databaseURL' : key.keyURL
})

def Calc_rate(in_data,out_data):
    in_result, out_result = []
    result_rate = []
    for i in range(0,7):#7일간의 데이터를 이용하여 7일의 예측데이터 가져오기
        result = model.predict(in_data[i])[3]#3이 최종결과값 (0~3까지있음)
        #나온 결과를 정규화
        scaler = MinMaxScaler()
        scaler.fit(result)
        in_result.append(scaler.transform(result))
    for i in range(0,7):#7일간의 데이터를 이용하여 7일의 예측데이터 가져오기
        result = model.predict(out_data[i])[3]#3이 최종결과값 (0~3까지있음)
        #나온 결과를 정규화
        scaler = MinMaxScaler()
        scaler.fit(result)
        out_result.append(scaler.transform(result))
        
    print(in_result)#debug

    for j in range(0,7):
        for i in range(0,19):#19시간(1일)간 들어온비율과나간비율을 계산(딕셔너리로 저장)
            if in_result[j][i] > out_result[j][i]:
                result_rate.append()
            else:
                result_rate.append()

db = firestore.client()
for k in db.collection("test").get():
    print(k.id, k.to_dict())
    input = k.to_dict()

#7일간의 검증x데이터를 이용하여 7일간의 예측데이터를 생성
#7일 후 예측y값과 실제 관측된 y값을 비교하여 loss가 일정치 이상 올라가면
#그동안 모은 데이터들을 이용하여 새로 학습 후 배포
#loss가 큰 변동이 없다면 학습하지 않고 7일간 모인 x데이터를 이용하여 predict만 수행함

model = keras.models.load_model('/data/_'+str(pipe.checkpoint_count)+'/mymodel')




#test
import matplotlib.pyplot as plt
#여러개출력으로 바꾸기
plt.plot(result)
plt.show()


