from tensorflow.keras.models import load_model
from normal_tools import save_data
import numpy as np
import os

np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    pre_input_data = np.load(file='../output/Processed data/splited/predict(0-1)_input_data.npy')
    pre_lable = np.load(file='../output/Processed data/splited/predict(0-1)_lable.npy')
    input_data = pre_input_data
    lable = pre_lable
    new_data = np.array(input_data)
    output_data = np.array(lable)
    X_test = new_data.reshape(len(new_data), 125000)
    y_test = output_data.reshape(len(output_data), 2500)
    model = load_model('../output/model/Dense_model_rates.h5')
    pred = model.predict(X_test)
    print(pred.shape)
    y_test = y_test.reshape(len(y_test), 2500)
    pred = pred.reshape(len(pred), 2500)

    save_data.save_data('../output/pred_result/pred_with_rate_two_dim.csv', pred)
    save_data.save_data('../output/pred_result/origin_with_rate_two_dim.csv', y_test)
