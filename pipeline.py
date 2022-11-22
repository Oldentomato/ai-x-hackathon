import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import LSTM as lt

checkpoint_count = 1

x_data = pd.read_csv("/data/dataset/b_xData")
y_data = pd.read_csv("/data/dataset/b_yData")

x_data = np.array(x_data)
y_data = np.array(y_data)

x_data = x_data.reshape(x_data.shape[0], 1, x_data.shape[1])

print(x_data)
print(y_data)

x_train,x_valid,y_train,y_valid = train_test_split(x_data,y_data,test_size=0.2)

x_train = lt.standarization(x_train)
y_train = lt.standarization(y_train)
x_valid = lt.standarization(x_valid)
y_valid = lt.standarization(y_valid)

print(x_valid)
print(y_valid)


latest = tf.train.latest_checkpoint('/data/checkpoint/_'+str(checkpoint_count)+'/')
model = lt.create_model()
model.load_weights(latest)
history = lt.run_model(model,lt.SetCallbacks(checkpoint_count))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'])

lt.multi_step_plot(x_valid[1], y_valid[1], model.predict(x_valid)[0])
model.save('/data/_2/mymodel')