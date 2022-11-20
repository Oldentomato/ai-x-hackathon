import pandas as pd
import matplotlib.pyplot as plt

bus_pos = pd.read_csv("dataset/buspos_info.csv", encoding='CP949')
bus_people = pd.read_csv("dataset/buspeople_info.csv", encoding='CP949')
road_info = pd.read_csv("dataset/road_info.csv", encoding='CP949')
bukchon = pd.read_csv("dataset/bukchon_pop.csv", encoding='CP949')
storepos = pd.read_csv("dataset/storepos_info.csv", encoding='CP949')


def Calculate_rate(dataframe):
    result_in_data = []
    result_out_data = []

    day_in_data = []
    day_out_data = []


    dataframe['start_date'] = pd.to_datetime(dataframe['측정 시작 시간'])
    dataframe['day'] = dataframe['start_date'].dt.day

    day_check = dataframe['day'][0]

    for i in range(0, dataframe['측정 시작 시간'].count() -1):
        if day_check != dataframe['day'][i]:
            result_in_data.append(list(day_in_data))
            result_out_data.append(list(day_out_data))
            day_in_data.clear()
            day_out_data.clear()
            day_check = dataframe['day'][i]
        else:
            day_in_data.append(dataframe['카메라 통과 인원 (IN)'][i] - dataframe['카메라 통과 인원 (IN)'][i+1])
            day_out_data.append(dataframe['카메라 통과 인원 (OUT)'][i] - dataframe['카메라 통과 인원 (OUT)'][i+1])



    return result_in_data, result_out_data


#북촌과 road_info 를 서로 비교하여 북촌의 길거리만 나오는 데이터프레임

#주소명을 바꿔주기위한 코드
bukchon.loc[bukchon[bukchon['주소']=='북촌로5가길 38'].index[:],['주소']] = ['북촌로5가길']
bukchon.loc[bukchon[bukchon['주소']=='계동길 69'].index[:],['주소']] = ['계동길']
bukchon.loc[bukchon[bukchon['주소']=='율곡로3길 50'].index[:],['주소']] = ['율곡로3길']
info = bukchon.drop_duplicates(['주소'])#중복되는 것들을 제외한 행만 출력 (3종류 (종류를 조회하기위함))

extracted_road_info = pd.DataFrame()

for i in range(0,3):
    extracted_road_info = pd.concat((extracted_road_info,road_info[road_info['노선명(도로명)'] == info['주소'][i]]))


result_in = []
result_out = []
b = bukchon.loc[bukchon[bukchon['주소']=='북촌로5가길'].index[:]]
b = b.reset_index(drop=True)
in_data, out_data = Calculate_rate(b)

in_data.reverse()
out_data.reverse()

result_in.append(in_data)
result_out.append(out_data)


fig, axes = plt.subplots(2,2)

axes[0][0].plot(out_data[0])
axes[0][1].plot(out_data[1])
axes[1][0].plot(out_data[5])
axes[1][1].plot(out_data[10])

plt.show()


#주소명에 따라 도로폭 칼럼 추가

# b['도로폭'] = 0
# g = bukchon.loc[bukchon[bukchon['주소']=='계동길'].index[:]]
# g['도로폭'] = 0
# y = bukchon.loc[bukchon[bukchon['주소']=='율곡로3길'].index[:]]
# y['도로폭'] = 1

# result_road = pd.concat([b,g,y])
# result_road = result_road[['주소','측정 시작 시간','측정 종료 시간','카메라 통과 인원 (IN)','카메라 통과 인원 (OUT)','도로폭']]
# print(result_road)


#북촌 주변 상권건물 데이터 가져오기
# bukchon_storepos = storepos.loc[storepos['시군구_코드']==11110][:]
# store_info = bukchon_storepos.drop_duplicates(['상권_구분_코드']) #A, D, R, U 4종류 나옴 (A = 골목상권 B = 발달상권 R = 전통시장 U = 관광특구)
# bukchon_storepos.loc[bukchon_storepos[bukchon_storepos['상권_구분_코드']=='A'].index[:],['상권_구분_코드']] = [0]
# bukchon_storepos.loc[bukchon_storepos[bukchon_storepos['상권_구분_코드']=='D'].index[:],['상권_구분_코드']] = [1]
# bukchon_storepos.loc[bukchon_storepos[bukchon_storepos['상권_구분_코드']=='R'].index[:],['상권_구분_코드']] = [2]
# bukchon_storepos.loc[bukchon_storepos[bukchon_storepos['상권_구분_코드']=='U'].index[:],['상권_구분_코드']] = [3]
# extracted_store = bukchon_storepos[['상권_구분_코드','엑스좌표_값','와이좌표_값']]
# print(extracted_store)

#주변 버스정보 데이터 가져오고 병합




#정제한 데이터들 병합 result_road, extracted_store, 
