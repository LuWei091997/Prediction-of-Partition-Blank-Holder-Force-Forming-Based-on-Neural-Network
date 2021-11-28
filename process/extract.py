from normal_tools import read_data
from normal_tools import save_data
import numpy as np


def get_max_and_min(arr):
    value_arr = []
    for i in arr:
        min_ = np.min(i)
        max_ = np.max(i)
        value_arr.append([min_, max_])
    return value_arr


if __name__ == '__main__':
    pred = read_data.readfile('../output/pred_result/pred_conv1d.csv')
    origin = read_data.readfile('../output/pred_result/origin_conv1d.csv')

    pred_min_max = get_max_and_min(pred)
    origin_min_max = get_max_and_min(origin)

    save_data.save_data('../output/pred_result/min and max/pred_conv1d_min_and_max.csv', pred_min_max)
    save_data.save_data('../output/pred_result/min and max/origin_conv1d_min_and_max.csv', origin_min_max)
