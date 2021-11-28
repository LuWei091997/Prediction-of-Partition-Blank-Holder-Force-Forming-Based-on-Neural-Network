import cv2
import numpy as np
from normal_tools import read_data
from normal_tools import save_data
from tools_for_image import get_the_number
from tools_for_image import read_all_pictures
from tools_for_image import get_color_list
from tools_for_image import creat_similar_image
from sklearn.model_selection import train_test_split
import time

np.set_printoptions(threshold=np.inf)


def get_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    color_dict = get_color_list.getColorList()
    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        if sum > maxsum:
            maxsum = sum
            color = d
    return color


def split_picture(img):
    img = img[60:860, 500:1300]
    vague_img = []
    step = 16
    range_1 = int(800 / step)
    for i in range(0, 800, step):
        for j in range(0, 800, step):
            img_part = img[i:i + step, j:j + step]
            vague_img.append(img_part)
    final_img = []
    for h in range(0, range_1):
        for j in range(0, range_1):
            if j == 0:
                final_img.append([])
            final_img[h].append(vague_img[range_1 * h + j])
    return final_img


def get_color_(image, thicken_red, thicken_blue, thicken_green, thicken_yellow, thicken_cyan_blue, thicken_orange):
    pic = []
    pic_w_col = []
    sa_color = []
    for i in range(len(image)):
        for j in range(len(image[i])):
            color = get_color(image[i][j])
            sa_color.append(color)
            if color == 'red':
                pic.append([i, j, thicken_red])
            elif color == 'red2':
                pic.append([i, j, thicken_red])
            elif color == 'blue':
                pic.append([i, j, thicken_blue])
            elif color == 'cyan_white':
                pic.append([i, j, 0])
            elif color == 'cyan':
                pic.append([i, j, thicken_cyan_blue])
            elif color == 'yellow':
                pic.append([i, j, thicken_yellow])
            elif color == 'orange':
                pic.append([i, j, thicken_orange])
            else:
                pic.append([i, j, 0])
        pic_w_col.append(sa_color)
        sa_color = []
    return pic, pic_w_col


def insert_param(data, Parameters):
    _param = []
    data_ = []
    for i in data:
        _param.extend(i[0:2])
        _param.extend(Parameters)
        _param.extend(i[2:])
        data_.append(_param)
        _param = []
    return data_


if __name__ == '__main__':
    path_ = '../data/picture'
    all_path = read_all_pictures.read_pictures(path_)
    param_path = '../data/data.csv'
    param = read_data.readfile(param_path)
    param = param[:, 0:-2]
    final_data = []
    thicken_rate_data = []
    for path in all_path:
        a = path.split('/', )
        b = a[4].split('.')
        num = (int(a[3]) - 1) * 100 + int(b[0])
        if 0 <= (num - 1) < 100:
            print(num)
            img = cv2.imread(path)
            print(path)
            start_time = time.time()
            parameters = param[num - 1]
            thicken_red = get_the_number.get_num('red', img)
            thicken_red = round((thicken_red - 1), 2)

            thicken_blue = get_the_number.get_num('blue', img)
            thicken_blue = round((thicken_blue - 1), 2)

            thicken_green = get_the_number.get_num('green', img)
            thicken_green = round((thicken_green - 1), 2)

            thicken_yellow = get_the_number.get_num('yellow', img)
            thicken_yellow = round((thicken_yellow - 1), 2)

            thicken_cyan_blue = get_the_number.get_num('cyan-blue', img)
            thicken_cyan_blue = round((thicken_cyan_blue - 1), 2)

            thicken_orange = get_the_number.get_num('orange', img)
            thicken_orange = round((thicken_orange - 1), 2)
            image = split_picture(img)
            data_with_thicken, color_name = get_color_(image, thicken_red, thicken_blue, thicken_green, thicken_yellow,
                                                       thicken_cyan_blue,
                                                       thicken_orange)
            sim_img = creat_similar_image.image_compose(color_name)
            path_im = '../output/sim_picture/' + str(num - 1) + '.jpg'
            cv2.imwrite(path_im, sim_img)
            data = insert_param(data_with_thicken, parameters)
            final_data.append(data)
            a = [abs(thicken_red / 100), abs(thicken_blue / 100)]
            thicken_rate_data.insert((num - 1), a)
            end_time = time.time()
            print('totle time：%fs' % (end_time - start_time))
            del img
            del start_time
            del end_time
            del sim_img
    save_data.save_data('../output/Processed data/extract_thicken.csv', thicken_rate_data)
    final_data = np.array(final_data)
    input_data = []
    lable = []
    for i in final_data:
        input_data.append(i[:, 0:-1])
        lable.append(i[:, -1])
    input_data = np.array(input_data)
    lable = np.array(lable)
    np.save(file='../output/Processed data/final_data_0-1_input_data.npy', arr=input_data)
    np.save(file='../output/Processed data/final_data_0-1_lable.npy', arr=lable)
    X_train, X_test, y_train, y_test = train_test_split(input_data, lable, test_size=0.2, shuffle=True,
                                                        random_state=None)
    np.save(file='../output/Processed data/splited/train(0-1)_input_data.npy', arr=X_train)
    np.save(file='../output/Processed data/splited/train(0-1)_lable.npy', arr=y_train)
    np.save(file='../output/Processed data/splited/predict(0-1)_input_data.npy', arr=X_test)
    np.save(file='../output/Processed data/splited/predict(0-1)_lable.npy', arr=y_test)
    print('finsh')
