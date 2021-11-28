
from tensorflow.keras.models import load_model
from normal_tools import save_data
import numpy as np
import os

np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    pre_input_data = np.load(file='../output/Processed data/splited/predict(0-1)_input_data.npy')
    pre_lable = np.load(file='../output/Processed data/splited/predict(0-1)_lable.npy')
    X_test = pre_input_data.reshape((20, 50, 50, 50))
    y_test = pre_lable.reshape((20, 50, 50))
    model = load_model('../output/model/Two_dim_model.h5')
    pred = model.predict(X_test)
    y_test = y_test.reshape(len(y_test), 2500)
    pred = pred.reshape(len(pred), 2500)
    save_data.save_data('../output/pred_result/pred_conv1d.csv', pred)
    save_data.save_data('../output/pred_result/origin_conv1d.csv', y_test)
