import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import LSTM as lt
import check_loss as chls


checkpoint_count = 1


dir_buckchon = "bukchon/b_"

x_in_data = pd.read_csv("/data/dataset/"+dir_buckchon+"depart_xData")
y_in_data = pd.read_csv("/data/dataset/"+dir_buckchon+"in_yData")
x_out_data = pd.read_csv("/data/dataset/"+dir_buckchon+"arr_xData")
y_out_data = pd.read_csv("/data/dataset/"+dir_buckchon+"out_yData")

x_in_data = np.array(x_in_data)
y_in_data = np.array(y_in_data)
x_out_data = np.array(x_out_data)
y_out_data = np.array(y_in_data)


x_in_data = x_in_data.reshape(x_in_data.shape[0], 1, x_in_data.shape[1])
x_out_data = x_out_data.reshape(x_out_data.shape[0], 1, x_out_data.shape[1])


print(x_in_data)
print(y_in_data)
print(x_out_data)
print(y_out_data)

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

print(x_in_train)
print(y_in_train)
print(x_out_train)
print(y_out_train)


indata_model = keras.models.load_model('/data/_'+str(checkpoint_count)+'/mymodel')
outdata_model = keras.models.load_model('/data/_'+str(checkpoint_count)+'/mymodel')
history_in = lt.run_model(indata_model,lt.SetCallbacks(checkpoint_count),x_in_train,y_in_train,x_in_valid,y_in_valid)
history_out = lt.run_model(outdata_model,lt.SetCallbacks(checkpoint_count),x_out_train,y_out_train,x_out_valid,y_out_valid)

fig, axes = plt.subplots(2,2)

axes[0][0].plot(history_in.history['loss'])
axes[0][1].plot(history_in.history['val_loss'])
axes[1][0].plot(history_out.history['loss'])
axes[1][1].plot(history_out.history['val_loss'])

plt.show()
lt.multi_step_plot(x_in_valid[1], y_in_valid[1], indata_model.predict(x_in_valid)[0])


indata_model.save('/data/_2/in/mymodel')
outdata_model.save('/data/_2/out/mymodel')