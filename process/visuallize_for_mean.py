from normal_tools import read_data
import random
import numpy as np
from normal_tools import save_data


def choose_(num_):
    nums = []
    for i in range(num_):
        num = random.randint(0, 200)
        nums.append(num)
        # print(nums)
    return nums


def visualize_(arr):
    element = []
    elements = []
    all_data = []
    for mean in arr:
        for i in range(len(mean)):
            for j in range(len(mean[i])):
                element = [i, j, mean[i][j]]
                elements.append(element)
        all_data.append(elements)
        elements = []
    return all_data


if __name__ == '__main__':
    path1 = '../output/pred_result/origin_dense.csv'
    path2 = '../output/pred_result/pred_dense.csv'
    origin_data = read_data.readfile(path1)
    pred_data = read_data.readfile(path2)
    print(len(origin_data), len(pred_data))

    needed_origin_data = []
    needed_pred_data = []
    for i in range(20):
        needed_origin_data.append(origin_data[i])  # 5*2500
        needed_pred_data.append(pred_data[i])  # 5*2500

    needed_origin_data = np.asarray(needed_origin_data)
    needed_pred_data = np.asarray(needed_pred_data)

    means = needed_pred_data - needed_origin_data

    origin_data = needed_origin_data.reshape((20, 50, 50))
    pred_data = needed_pred_data.reshape((20, 50, 50))
    means = means.reshape((20, 50, 50))

    origin_ = visualize_(origin_data)
    pred_ = visualize_(pred_data)
    mean_data = visualize_(means)
    print(len(origin_),len(pred_),len(mean_data))
    for data in mean_data:
        print('\r'+str(mean_data.index(data)),end='',flush=True)
        path = '../output/visualize/dense/mean_' + str(mean_data.index(data)) + '.csv'
        save_data.save_data(path, data)

    for or_ in origin_:
        print('\r' + str(origin_.index(or_)), end='', flush=True)
        path = '../output/visualize/dense/origin_' + str(origin_.index(or_)) + '.csv'
        save_data.save_data(path, or_)

    for pre in pred_:
        print('\r' + str(pred_.index(pre)), end='', flush=True)
        path = '../output/visualize/dense/pred_' + str(pred_.index(pre)) + '.csv'
        save_data.save_data(path, pre)
