from tensorflow import keras
import firebase_admin
from firebase_admin import credentials, firestore
import keyinfo as key
import numpy as np
import pipeline as pipe
from sklearn.preprocessing import MinMaxScaler

cred = credentials.Certificate("key/hackathon-11d72-firebase-adminsdk-7hnwq-45f6fa260b.json")
firebase_admin.initialize_app(cred,{
    'databaseURL' : key.keyURL
})

def Calc_rate(inmodel,outmodel,in_data,out_data,day):
    in_result = []
    out_result= []
    result_rate = []
    for i in range(0,day):#7일간의 데이터를 이용하여 7일의 예측데이터 가져오기
        result = inmodel.predict(in_data)[i]#validation 데이터 4개를 넣었기 때문에 4개의 결과값이 나옴
        #나온 결과를 정규화
        result = result.reshape(-1,1)
        scaler = MinMaxScaler()
        scaler.fit(result)
        result = scaler.transform(result)
        result = result.flatten()
        in_result.append(result)
    for i in range(0,day):
        result = outmodel.predict(out_data)[i]
        #나온 결과를 정규화
        result = result.reshape(-1,1)
        scaler = MinMaxScaler()
        scaler.fit(result)
        result = scaler.transform(result)
        result = result.flatten()
        out_result.append(result)
                                            

    for j in range(0,day):
        temp = []
        for i in range(0,18):#18시간(1일)간 들어온비율과나간비율을 계산(딕셔너리로 저장)
            if in_result[j][i] > out_result[j][i]:
                temp.append({'increase':in_result[j][i]-out_result[j][i]})
            elif in_result[j][i] < out_result[j][i]:
                temp.append({'decrease':out_result[j][i]-in_result[j][i]})
            else:
                temp.append({'not': 0})
        result_rate.append(temp)
                
    return result_rate

db = firestore.client()

# for k in db.collection("test").get():
#     print(k.id, k.to_dict())
    # input = k.to_dict()


#7일간의 검증x데이터를 이용하여 7일간의 예측데이터를 생성
#7일 후 예측y값과 실제 관측된 y값을 비교하여 loss가 일정치 이상 올라가면
#그동안 모은 데이터들을 이용하여 새로 학습 후 배포
#loss가 큰 변동이 없다면 학습하지 않고 7일간 모인 x데이터를 이용하여 predict만 수행함

<<<<<<< HEAD
region = 'gaedong'

indata_model = keras.models.load_model('models/'+region+'/_'+str(pipe.checkpoint_count)+'/in/mymodel.h5')
outdata_model = keras.models.load_model('models/'+region+'/_'+str(pipe.checkpoint_count)+'/out/mymodel.h5')

result = Calc_rate(indata_model,outdata_model,pipe.x_in_valid,pipe.x_out_valid,day=len(pipe.x_in_valid))
=======
indata_model = keras.models.load_model('/data/_'+str(pipe.checkpoint_count)+'/mymodel')
outdata_model = keras.models.load_model('/data/_'+str(pipe.checkpoint_count)+'/mymodel')
>>>>>>> c317dcc7badb5f7685cb53372e8c215045fbaaba

result = Calc_rate(indata_model,outdata_model,pipe.x_in_valid,pipe.x_out_valid,day=len(pipe.x_in_valid))

<<<<<<< HEAD
answer = input('do you permit to upload data? [y/n]')
if answer == 'y':
    doc_ref = db.collection(u'계동예측데이터').document(u'1주차')
    #결과를 db에 업로드
    data_convert = []
    for i in range(0,4):
        for j in range(0,18):
            data_convert.append({k:float(v) for k,v in result[i][j].items()})
        doc_ref.update({str(i+1)+'일': data_convert})
        data_convert.clear()
else:
    print(result)
=======
print(result)
>>>>>>> c317dcc7badb5f7685cb53372e8c215045fbaaba




