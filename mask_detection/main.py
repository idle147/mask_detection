from dataOpt import *
from train import *

if __name__ == '__main__':
    dataOpt = CDataOpt()
    # """
    # 进行训练
    # """
    # save_name = 'test_data.npy'
    # dataOpt.dataObtain('dataset/test/MASK', 'dataset/test/NO_MASK', save_name)
    # x_train, y_train = dataOpt.dataOpt(save_name)
    # modelOpt = CModelTrain()
    # CModelTrain.createModel(x_train, y_train)
    #
    # """
    # 进行预测
    # """
    # # 获取标签
    # save_name = 'test_data.npy'
    # dataOpt.dataObtain('dataset/test/MASK', 'dataset/test/NO_MASK', save_name)
    # x_test, y_test = dataOpt.dataOpt(save_name)
    # # 模型读取
    # modelOpt = CModelTrain()
    # network = modelOpt.modelStructure()
    # model = modelOpt.readModel(network)
    # # 模型使用
    # predict_test = []
    # for i in x_test:
    #     i = i.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 3))
    #     predict_test.append(modelOpt.usingModel(model, i))
    # modelOpt.getScore(y_test, predict_test)
