import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import LSTM_tensor as lt


def data_reshape(data):
    result = []
    date_kind = data.drop_duplicates(['0'])
    date_kind = date_kind.reset_index(drop=True)
    date_kind_count = len(date_kind)
    for i in range(0,date_kind_count):
        temp = data.loc[data[data['0']==date_kind['0'][i]].index[:]]
        temp.drop(['0','1'],axis=1,inplace=True)
        temp = np.array(temp)
        result.append(temp)
    result = np.array(result)
    return result 


checkpoint_count = 1


dir = "gaedong"

x_in_data = pd.read_csv("resultdata/"+dir+"/depart_xData")
y_in_data = pd.read_csv("resultdata/"+dir+"/in_yData")
x_out_data = pd.read_csv("resultdata/"+dir+"/arr_xData")
y_out_data = pd.read_csv("resultdata/"+dir+"/out_yData")


x_in_data = data_reshape(x_in_data)
y_in_data = np.array(y_in_data)
x_out_data = data_reshape(x_out_data)
y_out_data = np.array(y_in_data)



# x_in_data = x_in_data.reshape(x_in_data.shape[0], date_kind_count, x_in_data.shape[1])
# x_out_data = x_out_data.reshape(x_out_data.shape[0], date_kind_count, x_out_data.shape[1])


print(x_in_data.shape)
print(y_in_data.shape)
print(x_out_data.shape)
print(y_out_data.shape)


x_in_train,x_in_valid,y_in_train,y_in_valid = train_test_split(x_in_data,y_in_data,test_size=0.2)
x_out_train,x_out_valid,y_out_train,y_out_valid = train_test_split(x_out_data,y_out_data,test_size=0.2)

x_in_train = lt.standarization(x_in_train)
y_in_train = lt.standarization(y_in_train)
x_in_valid = lt.standarization(x_in_valid)
y_in_valid = lt.standarization(y_in_valid)

x_out_train = lt.standarization(x_out_train)
y_out_train = lt.standarization(y_out_train)
x_out_valid = lt.standarization(x_out_valid)
y_out_valid = lt.standarization(y_out_valid)

# print(x_in_train)
# print(y_in_train)
# print(x_out_train)
# print(y_out_train)


indata_model = lt.create_model(x_in_train,64,32,0.05)
outdata_model = lt.create_model(x_out_train,64,32,0.2)


history_in = lt.RunAndSave_model(indata_model,'gaedong','in',lt.SetCallbacks(checkpoint_count),x_in_train,y_in_train,x_in_valid,y_in_valid)
history_out = lt.RunAndSave_model(outdata_model,'gaedong','out',lt.SetCallbacks(checkpoint_count),x_out_train,y_out_train,x_out_valid,y_out_valid)



fig, axes = plt.subplots(2,2)

axes[0][0].plot(history_in.history['loss'])
axes[0][1].plot(history_in.history['val_loss'])
axes[1][0].plot(history_out.history['loss'])
axes[1][1].plot(history_out.history['val_loss'])

plt.show()
lt.multi_step_plot(x_in_valid[0], y_in_valid[0], indata_model.predict(x_in_valid)[0])