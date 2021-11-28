import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf

np.set_printoptions(threshold=np.inf)
optimizer = tf.keras.optimizers.RMSprop(0.001)


def create_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(125000,), activation='softplus'))
    model.add(Dense(256, activation='softplus'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(256, activation='softplus'))
    model.add(Dense(2500, name='output_layer', activation='linear'))
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    model.summary()
    return model


if __name__ == '__main__':
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input_data_train = np.load(file='../output/Processed data/splited/train(0-1)_input_data.npy')
    lable_train = np.load(file='../output/Processed data/splited/train(0-1)_lable.npy')
    input_data_pred = np.load(file='../output/Processed data/splited/predict(0-1)_input_data.npy')
    lable_pred = np.load(file='../output/Processed data/splited/predict(0-1)_lable.npy')
    X_train = input_data_train.reshape(len(input_data_train), 125000)
    y_train = lable_train.reshape(len(lable_train), 2500)
    X_test = input_data_pred.reshape(len(input_data_pred), 125000)
    y_test = lable_pred.reshape(len(lable_pred), 2500)
    model = create_model()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=2)
    model_path = '../output/model/Dense_model.h5'
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
    print('totle time%fs' % (end_time - start_time))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    np_loss = np.array(loss).reshape(1, len(loss))
    np_val_loss = np.array(val_loss).reshape(1, len(val_loss))
    mae = np.array(mae).reshape(1, len(mae))
    np_val_mae = np.array(val_mae).reshape(1, len(val_mae))
    # 输出位置 loss, mae, val_loss, val_mae
    loss_data = np.concatenate([np_loss, mae, np_val_loss, np_val_mae], axis=0)
    np.savetxt('../output/model/train_detail/loss_data_dense.txt', loss_data)
