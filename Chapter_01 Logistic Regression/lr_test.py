# coding:UTF-8
'''
Date:20160901
@author: zhaozhiyong
'''
import numpy as np
from lr_train import sig

def load_weight(w):
    '''导入LR模型
    input:  w(string)权重所在的文件位置
    output: np.mat(w)(mat)权重的矩阵
    '''
    f = open(w)
    w = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        w_tmp = []
        for x in lines:
            w_tmp.append(float(x))
        w.append(w_tmp)    
    f.close()
    return np.mat(w)

def load_data(file_name, n):
    '''导入测试数据
    input:  file_name(string)测试集的位置
            n(int)特征的个数
    output: np.mat(feature_data)(mat)测试集的特征
    '''
    f = open(file_name)
    feature_data = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        # print lines[2]
        if len(lines) <> n - 1:
            continue
        feature_tmp.append(1)
        for x in lines:
            # print x
            feature_tmp.append(float(x))
        feature_data.append(feature_tmp)
    f.close()
    return np.mat(feature_data)

def predict(data, w):
    '''对测试数据进行预测
    input:  data(mat)测试数据的特征
            w(mat)模型的参数
    output: h(mat)最终的预测结果
    '''
    h = sig(data * w.T)#sig
    m = np.shape(h)[0]
    for i in xrange(m):
        if h[i, 0] < 0.5:
            h[i, 0] = 0.0
        else:
            h[i, 0] = 1.0
    return h

def save_result(file_name, result):
    '''保存最终的预测结果
    input:  file_name(string):预测结果保存的文件名
            result(mat):预测的结果
    '''
    m = np.shape(result)[0]
    #输出预测结果到文件
    tmp = []
    for i in xrange(m):
        tmp.append(str(result[i, 0]))
    f_result = open(file_name, "w")
    f_result.write("\t".join(tmp))
    f_result.close()    

if __name__ == "__main__":
    # 1、导入LR模型
    print "---------- 1.load model ------------"
    w = load_weight("weights")
    n = np.shape(w)[1]
    # 2、导入测试数据
    print "---------- 2.load data ------------"
    testData = load_data("test_data", n)
    # 3、对测试数据进行预测
    print "---------- 3.get prediction ------------"
    h = predict(testData, w)#进行预测
    # 4、保存最终的预测结果
    print "---------- 4.save prediction ------------"
    save_result("result", h)
    
