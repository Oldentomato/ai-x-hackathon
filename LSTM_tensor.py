import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

def run_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            

def standarization(data): #표준화 (정규화는 효과가 안좋았음)
    data_mean = data[:].mean()
    data_std = data[:].std()
    data = (data-data_mean)/data_std
    return data

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(loss))
    
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    
    plt.show()

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12,6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    
    plt.plot(num_in, np.array(history[:,1]), label='History')
    plt.plot(np.arange(num_out)/19, np.array(true_future),'bo', label='True Future')
    
    if prediction.any():
        plt.plot(np.arange(num_out)/19, np.array(prediction), 'ro', label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

def create_time_steps(length):
    return list(range(-length,0))

x_data = pd.read_csv("/data/dataset/b_xData")
y_data = pd.read_csv("/data/dataset/b_yData")

x_data = np.array(x_data)
y_data = np.array(y_data)

x_data = x_data.reshape(x_data.shape[0], 1, x_data.shape[1])

print(x_data)
print(y_data)


x_train,x_valid,y_train,y_valid = train_test_split(x_data,y_data,test_size=0.2)

x_train = standarization(x_train)
y_train = standarization(y_train)
x_valid = standarization(x_valid)
y_valid = standarization(y_valid)

print(x_valid)
print(y_valid)


def SetCallbacks(checkpoint_count):
    checkpoint = ModelCheckpoint(
        '/data/checkpoint/_'+str(checkpoint_count+1)+'/epoch_{epoch:03d}.ckpt',
        monitor = 'val_loss',
        save_best_only = True
    )

    callbacks = [checkpoint]
    return callbacks

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=x_train.shape[-2:]))
    model.add(tf.keras.layers.LSTM(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Dense(19))#18시간(1일) 데이터

    model.summary()

    model.compile(optimizer="adam", loss='mean_squared_error')
    
    return model
    

def run_model(model,callback):
    with tf.device("/gpu:0"):
        history = model.fit(x_train,y_train,
                                       epochs=100,
                                       validation_data=(x_valid,y_valid),
                                       batch_size=64,
                                       callbacks = callback,
                                        shuffle=True
                                      )
    return history

