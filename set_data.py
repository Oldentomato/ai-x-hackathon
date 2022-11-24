import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

bus_pos = pd.read_csv("dataset/buspos_info.csv", encoding='CP949')
bus_people = pd.read_csv("dataset/buspeople_info.csv", encoding='CP949')
road_info = pd.read_csv("dataset/road_info.csv", encoding='CP949')
bukchon = pd.read_csv("dataset/bukchon_pop.csv", encoding='CP949')
storepos = pd.read_csv("dataset/storepos_info.csv", encoding='CP949')


def Data_Processing(bus_station,in_data,out_data):
    near_arrive_bus, near_depart_bus = extract_businfo(bus_station)
    near_arrive_bus.drop(['기준_날짜'],axis=1, inplace=True)
    near_depart_bus.drop(['기준_날짜'],axis=1, inplace=True)

    near_arrive_bus = near_arrive_bus.values.tolist()
    near_depart_bus = near_depart_bus.values.tolist()

    near_arrive_bus.reverse()
    near_depart_bus.reverse()


    in_data = pd.DataFrame(in_data)
    out_data = pd.DataFrame(out_data)
    near_arrive_bus = pd.DataFrame(near_arrive_bus)
    near_depart_bus = pd.DataFrame(near_depart_bus)

    # in_data = in_data.drop([9,18])
    # out_data = out_data.drop([9,18])

    in_data = in_data.fillna(method='ffill')
    out_data = out_data.fillna(method='ffill')
    near_arrive_bus = near_arrive_bus.fillna(method='bfill')
    near_depart_bus = near_depart_bus.fillna(method='bfill')

    in_data.reset_index(inplace=True,drop=True)
    out_data.reset_index(inplace=True,drop=True)



    return near_arrive_bus,near_depart_bus,in_data,out_data


def SaveData(dir,data_1,data_2,data_3,data_4):
    """
    Pandas to csv Saving
        Args:
            dir `string`: SaveRootDirectoryURL
            data_1 `pandas`: arrive_bus_station
            data_2 `pandas`: depart_bus_station
            data_3 `pandas`: in_data
            data_4 `pandas`: out_data
        Returns:
            None
    """
    data_1.to_csv("resultdata/"+dir+"_arr_xData",header=True,index=False)
    data_2.to_csv("resultdata/"+dir+"_depart_xData",header=True,index=False)

    data_3.to_csv("resultdata/"+dir+"_in_yData",header=True,index=False)
    data_4.to_csv("resultdata/"+dir+"_out_yData",header=True,index=False)


def Calculate_rate(dataframe):
    result_in_data = []
    result_out_data = []

    day_in_data = []
    day_out_data = []


    dataframe['start_date'] = pd.to_datetime(dataframe['측정 시작 시간'])
    dataframe['day'] = dataframe['start_date'].dt.day
    dataframe['month'] = dataframe['start_date'].dt.month
    dataframe['hour'] = dataframe['start_date'].dt.hour
    dataframe['minute'] = dataframe['start_date'].dt.minute

    #1시간간격으로 하기위해 0분(정각) 이외의 시간 다 없애기
    dataframe = dataframe.loc[dataframe['minute'] == 0][:]
    dataframe = dataframe.reset_index(drop=True)
    day_check = dataframe['day'][0]

    for i in range(0, dataframe['측정 시작 시간'].count() -1):
        if day_check != dataframe['day'][i] and day_in_data:
            day_in_data.pop()
            day_out_data.pop()
            result_in_data.append(list(day_in_data))
            result_out_data.append(list(day_out_data))
            day_in_data.clear()
            day_out_data.clear()
            day_check = dataframe['day'][i]

        if (dataframe['month'][i] == 10 and dataframe['day'][i] >= 27) or (dataframe['month'][i] == 11 and dataframe['day'][i] <= 13):
            if dataframe['hour'][i] >= 5:
                day_in_data.append(dataframe['카메라 통과 인원 (IN)'][i] - dataframe['카메라 통과 인원 (IN)'][i+1])
                day_out_data.append(dataframe['카메라 통과 인원 (OUT)'][i] - dataframe['카메라 통과 인원 (OUT)'][i+1])


    return result_in_data, result_out_data

#지정된 정류장들을 탐색 후 시간별 재차인원 구하기
def extract_businfo(bus_station):
    for i in range(0,len(bus_station)):
        temp = bus_pos.loc[bus_pos['정류장_명칭'] == bus_station[i]][:]
    for j in temp['정류장_ID']:
        bus_arrive_result = bus_people.loc[bus_people['도착_정류장_ID'] == j][:]
        bus_depart_result = bus_people.loc[bus_people['출발_정류장_ID'] == j][:]
    
    #해당 버스의 날짜를 추출
    bus_arrive_result = bus_arrive_result.iloc[:,[0,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]]
    bus_depart_result = bus_depart_result.iloc[:,[0,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]]
    return bus_arrive_result, bus_depart_result


#북촌과 road_info 를 서로 비교하여 북촌의 길거리만 나오는 데이터프레임

#주소명을 바꿔주기위한 코드
bukchon.loc[bukchon[bukchon['주소']=='북촌로5가길 38'].index[:],['주소']] = ['북촌로5가길']
bukchon.loc[bukchon[bukchon['주소']=='계동길 69'].index[:],['주소']] = ['계동길']
bukchon.loc[bukchon[bukchon['주소']=='율곡로3길 50'].index[:],['주소']] = ['율곡로3길']
info = bukchon.drop_duplicates(['주소'])#중복되는 것들을 제외한 행만 출력 (3종류 (종류를 조회하기위함))

extracted_road_info = pd.DataFrame()

for i in range(0,3):
    extracted_road_info = pd.concat((extracted_road_info,road_info[road_info['노선명(도로명)'] == info['주소'][i]]))


b = bukchon.loc[bukchon[bukchon['주소']=='북촌로5가길'].index[:]]
b = b.reset_index(drop=True)
b_in_data, b_out_data = Calculate_rate(b)
b_in_data.reverse()
b_out_data.reverse()

g = bukchon.loc[bukchon[bukchon['주소']=='계동길'].index[:]]
g = g.reset_index(drop=True)
g_in_data, g_out_data = Calculate_rate(g)
g_in_data.reverse()
g_out_data.reverse()

y = bukchon.loc[bukchon[bukchon['주소']=='율곡로3길'].index[:]]
y = y.reset_index(drop=True)
y_in_data, y_out_data = Calculate_rate(g)
y_in_data.reverse()
y_out_data.reverse()


#길이 51(일)
# print(len(b_out_data))

#시각확인용
# fig, axes = plt.subplots(2,2)

# axes[0][0].plot(b_out_data[1])
# axes[0][1].plot(b_out_data[10])
# axes[1][0].plot(b_out_data[3])
# axes[1][1].plot(b_out_data[11])

# plt.show()





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
b_bus_station = ['삼청파출소','정독도서관','경복궁.국립민속박물관','국립민속박물관']
g_bus_station = ['중앙중고','원서고개','사우디대사관','사우디대사관앞.경남빌라']
y_bus_station = ['덕성여중고','인사동.북촌','안국역.서울공예박물관','안국역.인사동','정독도서관','경복궁.국립민속박물관','국립민속박물관','법련사','경복궁','안국역','안국동']


b_arrive_bus_station,b_depart_bus_station,b_in_data,b_out_data = Data_Processing(b_bus_station,b_in_data,b_out_data)
g_arrive_bus_station,g_depart_bus_station,g_in_data,g_out_data = Data_Processing(g_bus_station,g_in_data,g_out_data)
y_arrive_bus_station,y_depart_bus_station,y_in_data,y_out_data = Data_Processing(y_bus_station,y_in_data,y_out_data)

print(g_arrive_bus_station)
print(g_depart_bus_station)
print(g_in_data)
print(g_out_data)

# fig, axes = plt.subplots(2,2)

# axes[0][0].plot(g_arrive_bus_station)
# axes[0][1].plot(g_depart_bus_station)
# axes[1][0].plot(g_in_data)
# axes[1][1].plot(g_out_data)

# plt.show()

#y도로폭 = 1
# b_in_data = np.array(b_in_data)
# b_in_data = b_in_data.flatten()



# SaveData("bukchon/b",b_arrive_bus_station,b_depart_bus_station,b_in_data,b_out_data)
# SaveData("gaedong/g",g_arrive_bus_station,g_depart_bus_station,g_in_data,g_out_data)
# SaveData("yulgok/y",y_arrive_bus_station,y_depart_bus_station,y_in_data,y_out_data)

print("done")


#정제한 데이터들 병합
