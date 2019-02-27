# coding:UTF-8

import numpy as np
from linear_regression_train import load_data

def lwlr(feature, label, k):
    '''局部加权线性回归
    input:  feature(mat):特征
            label(mat):标签
            k(int):核函数的系数
    output: predict(mat):最终的结果
    '''
    m = np.shape(feature)[0]
    predict = np.zeros(m)
    weights = np.mat(np.eye(m))
    for i in xrange(m):
        for j in xrange(m):
            diff = feature[i, ] - feature[j, ]
            weights[j,j] = np.exp(diff * diff.T / (-2.0 * k ** 2))
        xTx = feature.T * (weights * feature)
        ws = xTx.I * (feature.T * (weights * label))
        predict[i] = feature[i, ] * ws
    return predict

if __name__ == "__main__":
    # 1、导入数据集
    feature, label = load_data("data.txt")
    predict = lwlr(feature, label, 0.002)
    m = np.shape(predict)[0]
    for i in xrange(m):
        print feature[i, 1], predict[i]
