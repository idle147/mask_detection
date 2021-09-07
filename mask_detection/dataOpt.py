import os
import numpy as np
import pandas as pd
import cv2
from tflearn.data_utils import shuffle
from tqdm import tqdm
from parameter import *


class CDataOpt:
    """
        进行基本的数据操作
    """

    # def readTxt(self, file_name):
    #     """
    #     读取txt文件
    #     :param file_name: txt文件名
    #     :return:
    #     """
    #     # 读取txt文件的每一行
    #     f = open(file_name)
    #     line_list = []
    #     line = f.readline()
    #     line_list.append(line)
    #     # 添加append
    #     while line:
    #         line = f.readline()
    #         line_list.append(line)
    #     f.close()
    #     return line_list
    #
    # def dataClear(self, directory_name, positive_path, negative_path):
    #     """
    #     数据清洗与标记
    #     :return:
    #     """
    #     # 标签
    #     positive_label = np.zeros(2)
    #     negative_label = np.zeros(2)
    #     # 样本数量
    #     positive = 0
    #     negative = 0
    #     # 读取文件夹下所有的图片文件
    #     for filename in tqdm(os.listdir(r"./" + directory_name)):
    #         # 读取图片文件
    #         img_path = directory_name + "/" + filename
    #         img = cv2.imread(img_path)
    #
    #         # 读取txt文件
    #         txt_path = img_path.split('.', 1)[0] + '.txt'
    #         line_list = self.readTxt(txt_path)
    #
    #         # 循环截取图片
    #         for i in line_list:
    #             temp = i.split(' ')
    #             y0 = int(temp[1])
    #             y1 = int(temp[2])
    #             x0 = int(temp[3])
    #             x1 = int(temp[4])
    #             cropped = img[y0:y1, x0:x1]  # 裁剪坐标为[y0:y1, x0:x1]
    #
    #             if temp[0] == '0':
    #                 name_path = negative_path + '/' + str(negative) + '.jpg'
    #                 cv2.imwrite(name_path, cropped)  # 保存文件
    #                 negative += 1
    #                 negative_label.append(0, name_path)
    #                 print("截取人脸图片%d.jpg", negative)
    #             elif temp[0] == '1':
    #                 name_path = positive_path + '/' + str(positive) + '.jpg'
    #                 cv2.imwrite(name_path, cropped)
    #                 positive += 1
    #                 positive_label.append(1, name_path)
    #                 print("截取带口罩图片%d.jpg", positive)
    #             else:
    #                 print("无法识别正负样本", positive)
    #     # 将标签文件保存成CSV
    #     data1 = pd.DataFrame(positive_label)
    #     data1.to_csv('positive_label.csv')
    #     data2 = pd.DataFrame(negative_label)
    #     data2.to_csv('negative_label.csv')

    def dataObtain(self, positive_path, negative_path, save_name):
        """
        获取数据标签
        :param positive_path: 正样本路径
        :param negative_path: 负样本路径
        :return: 保存成np
        """
        # 数量
        train_data = []

        # 读取正样本文件夹下所有的图片文件
        for filename in tqdm(os.listdir(r"./" + positive_path)):
            # 读取文件
            img = cv2.imread(positive_path + '/' + filename)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            # 加入数组
            train_data.append([np.array(img), np.array([1, 0])])
        # 读取负样本文件夹下所有的图片文件
        for filename in tqdm(os.listdir(r"./" + negative_path)):
            # 读取文件
            img = cv2.imread(negative_path + '/' + filename)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            # 加入数组
            train_data.append([np.array(img), np.array([0, 1])])
        shuffle(train_data)
        np.save(save_name, train_data)

    @staticmethod
    def dataOpt(path):
        # 数量
        train_data = np.load(path, allow_pickle=True)
        x_train = np.array([i[0] for i in train_data]).reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 3))
        # 标签
        y_train = [i[1] for i in train_data]
        # 获取标签数据
        return x_train, y_train

    @staticmethod
    def imgClassify(img_path):
        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_arr is not None:
            # img = cv2.resize(img_arr, (IMAGE_SIZE, IMAGE_SIZE))
            # img = img.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
            return img_arr
        else:
            return 0
