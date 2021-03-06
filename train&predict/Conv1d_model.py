import numpy as np
from keras.layers import Dense, Conv2D, Flatten, Conv1D, MaxPooling1D, LSTM, Dropout
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import matplotlib.pyplot as plt
from normal_tools import save_data
import os
import time
import tensorflow as tf

np.set_printoptions(threshold=np.inf)
optimizer = tf.keras.optimizers.RMSprop(0.001)


def create_model():
    model = Sequential()
    model.add(Conv1D(name='input_layer', filters=40, kernel_size=1, activation='tanh', input_shape=(50, 50, 50)))
    model.add(Conv1D(name='layer_1', filters=30, kernel_size=1, activation='sigmoid'))
    model.add(Conv1D(name='layer_2', filters=30, kernel_size=1, activation='tanh'))
    model.add(Conv1D(name='layer_3', filters=10, kernel_size=1, activation='linear'))
    model.add(Conv1D(name='layer_5', filters=10, kernel_size=1, activation='tanh'))
    model.add(Conv1D(name='layer_7', filters=3, kernel_size=1, activation='sigmoid'))
    model.add(Conv1D(name='layer_8', filters=3, kernel_size=1, activation='tanh'))
    model.add(Conv1D(name='layer_9', filters=3, kernel_size=1, activation='linear'))
    model.add(Dense(1, name='output_layer', activation='linear'))
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    model.summary()
    return model


if __name__ == '__main__':
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    input_data_train = np.load(file='../output/Processed data/splited/train(0-1)_input_data.npy')
    lable_train = np.load(file='../output/Processed data/splited/train(0-1)_lable.npy')
    input_data_pred = np.load(file='../output/Processed data/splited/predict(0-1)_input_data.npy')
    lable_pred = np.load(file='../output/Processed data/splited/predict(0-1)_lable.npy')
    input_data = input_data_train.reshape((800, 50, 50, 50))
    lable = lable_train.reshape((800, 50, 50))
    X_test = input_data_pred.reshape((200, 50, 50, 50))
    y_test = lable_pred.reshape((200, 50, 50))
    model = create_model()
    history = model.fit(input_data, lable, validation_data=(X_test, y_test), epochs=5000, batch_size=1)
    model_path = '../output/model/Two_dim_model__.h5'
    model.save(model_path)

    plt.plot()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot()
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    end_time = time.time()
    print('totle time %fs' % (end_time - start_time))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']

    save_data.save_data('../output/model/train_detail/loss.csv', loss)
    save_data.save_data('../output/model/train_detail/val_loss.csv', val_loss)
    save_data.save_data('../output/model/train_detail/mae.csv', mae)
    save_data.save_data('../output/model/train_detail/val_mae.csv', val_mae)

