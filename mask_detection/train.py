"""
    图片大小 50
    分类类数: 2类
    输入层 卷积层 池化层 激活层 全连接层 输出层

    流程:
        一、模型搭建
        二、数据处理
        三、模型预测
"""
import numpy as np
import tflearn
from parameter import *
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization


class CModelTrain:
    """
        模型的生成
    """

    def __init__(self):
        """
        构造函数, 进行模型的训练
        """

    def modelStructure(self):
        # 输入层 placeholder
        conv_input = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input")

        """ 第一层 卷积池化层"""
        # incoming, nb_filter输出高度, filter_size 卷积核大小
        network = conv_2d(conv_input, 96, 11, strides=4, activation="relu")  # 卷积
        network = max_pool_2d(network, 3, strides=2)  # 池化
        network = local_response_normalization(network)

        """ 第二层 卷积池化层"""
        network = conv_2d(network, 256, 5, activation="relu")  # 卷积
        network = max_pool_2d(network, 3, strides=2)  # 池化
        network = local_response_normalization(network)

        """ 第三层 三重卷积层"""
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 384, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)

        """ 第四层 双重全连接层 """
        # incoming, n_units, activation='linear'
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)
        network = fully_connected(network, 4096, activation='tanh')
        network = dropout(network, 0.5)

        """ 第五层 输出层 """
        network = fully_connected(network, 2, activation="softmax")  # 全连接层

        # 构建损失函数核优化器
        network = regression(network, placeholder='default', optimizer='momentum',  # 优化器
                             loss='categorical_crossentropy',
                             learning_rate=LEARNING_RATE)
        return network

    @staticmethod
    def createModel(network, x_train, y_train):
        # 搭建AlexNet网络
        tflearn.init_graph(gpu_memory_fraction=1)

        # 创建模型
        model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                            max_checkpoints=1, tensorboard_verbose=2)
        # 训练模型
        model.fit(x_train, y_train, n_epoch=EPOLL_TIME, validation_set=0.1,
                  show_metric=True,  # 显示日志
                  snapshot_epoch=False,
                  snapshot_step=MODEL_SNAP_TIME,  # 跑N次进行快照
                  run_id='alexnet', batch_size=BATCH_SIZE)
        # 训练模型
        model.save(MODEL_NAME)
        print("模型训练完成")

    @staticmethod
    def readModel(network):
        model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                            max_checkpoints=1, tensorboard_verbose=2)
        model.load(MODEL_NAME, weights_only=True)
        return model

    @staticmethod
    def usingModel(model, img):
        num = model.predict(img)  # 获取概率
        classify = np.argmax(num)  # 获取标签
        if classify == 0:
            print("No_MASK")
            return np.array([1, 0])
        elif classify == 1:
            print("MASK")
            return np.array([0, 1])
        else:
            print("Error: Masks cannot be identified")
            return np.array([0, 0])

    @staticmethod
    def getScore(y_test, predict_test):
        correct = 0
        error = 0
        for i in range(len(y_test)):
            print(y_test[i])
            if (y_test[i] == predict_test[i]).all():
                correct += 1
            else:
                error += 1
        res = correct / len(y_test)
        print('Test accuracy:%.2f%%' % res)
