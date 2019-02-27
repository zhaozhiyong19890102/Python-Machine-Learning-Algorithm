# coding:UTF-8

import numpy as np
import random as rd

def load_weights(weights_path):
    '''导入训练好的Softmax模型
    input:  weights_path(string)权重的存储位置
    output: weights(mat)将权重存到矩阵中
            m(int)权重的行数
            n(int)权重的列数
    '''
    f = open(weights_path)
    w = []
    for line in f.readlines():
        w_tmp = []
        lines = line.strip().split("\t")
        for x in lines:
            w_tmp.append(float(x))
        w.append(w_tmp)
    f.close()
    weights = np.mat(w)
    m, n = np.shape(weights)
    return weights, m, n

def load_data(num, m):
    '''导入测试数据
    input:  num(int)生成的测试样本的个数
            m(int)样本的维数
    output: testDataSet(mat)生成测试样本
    '''
    testDataSet = np.mat(np.ones((num, m)))
    for i in xrange(num):
        testDataSet[i, 1] = rd.random() * 6 - 3#随机生成[-3,3]之间的随机数
        testDataSet[i, 2] = rd.random() * 15#随机生成[0,15]之间是的随机数
    return testDataSet

def predict(test_data, weights):
    '''利用训练好的Softmax模型对测试数据进行预测
    input:  test_data(mat)测试数据的特征
            weights(mat)模型的权重
    output: h.argmax(axis=1)所属的类别
    '''
    h = test_data * weights
    return h.argmax(axis=1)#获得所属的类别

def save_result(file_name, result):
    '''保存最终的预测结果
    input:  file_name(string):保存最终结果的文件名
            result(mat):最终的预测结果
    '''
    f_result = open(file_name, "w")
    m = np.shape(result)[0]
    for i in xrange(m):
        f_result.write(str(result[i, 0]) + "\n")
    f_result.close()
    

if __name__ == "__main__":
    # 1、导入Softmax模型
    print "---------- 1.load model ------------"
    w, m , n = load_weights("weights")
    # 2、导入测试数据
    print "---------- 2.load data ------------"
    test_data = load_data(4000, m)
    # 3、利用训练好的Softmax模型对测试数据进行预测
    print "---------- 3.get Prediction ------------"
    result = predict(test_data, w)
    # 4、保存最终的预测结果
    print "---------- 4.save prediction ------------"
    save_result("result", result)
