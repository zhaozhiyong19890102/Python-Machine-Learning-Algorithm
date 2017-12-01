# coding:UTF-8
'''
Date:20160831
@author: zhaozhiyong
'''
import numpy as np
from random import normalvariate  # 正态分布

def loadDataSet(data):
    '''导入训练数据
    input:  data(string)训练数据
    output: dataMat(list)特征
            labelMat(list)标签
    '''
    dataMat = []
    labelMat = []   
    fr = open(data)  # 打开文件  
    for line in fr.readlines():
        lines = line.strip().split("\t")
        lineArr = []
        
        for i in xrange(len(lines) - 1):
            lineArr.append(float(lines[i]))
        dataMat.append(lineArr)
        
        labelMat.append(float(lines[-1]) * 2 - 1)  # 转换成{-1,1}
    fr.close()
    return dataMat, labelMat

def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))

def initialize_v(n, k):
    '''初始化交叉项
    input:  n(int)特征的个数
            k(int)FM模型的超参数
    output: v(mat):交叉项的系数权重
    '''
    v = np.mat(np.zeros((n, k)))
    
    for i in xrange(n):
        for j in xrange(k):
            # 利用正态分布生成每一个权重
            v[i, j] = normalvariate(0, 0.2)
    return v

def stocGradAscent(dataMatrix, classLabels, k, max_iter, alpha):
    '''利用随机梯度下降法训练FM模型
    input:  dataMatrix(mat)特征
            classLabels(mat)标签
            k(int)v的维数
            max_iter(int)最大迭代次数
            alpha(float)学习率
    output: w0(float),w(mat),v(mat):权重
    '''
    m, n = np.shape(dataMatrix)
    # 1、初始化参数
    w = np.zeros((n, 1))  # 其中n是特征的个数
    w0 = 0  # 偏置项
    v = initialize_v(n, k)  # 初始化V
    
    # 2、训练
    for it in xrange(max_iter):
        for x in xrange(m):  # 随机优化，对每一个样本而言的
            inter_1 = dataMatrix[x] * v
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * \
             np.multiply(v, v)  # multiply对应元素相乘
            # 完成交叉项
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
            p = w0 + dataMatrix[x] * w + interaction  # 计算预测的输出
            loss = sigmoid(classLabels[x] * p[0, 0]) - 1
        
            w0 = w0 - alpha * loss * classLabels[x]
            for i in xrange(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]
                    
                    for j in xrange(k):
                        v[i, j] = v[i, j] - alpha * loss * classLabels[x] * \
                        (dataMatrix[x, i] * inter_1[0, j] -\
                          v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
        
        # 计算损失函数的值
        if it % 1000 == 0:
            print "\t------- iter: ", it, " , cost: ", \
            getCost(getPrediction(np.mat(dataMatrix), w0, w, v), classLabels)
    
    # 3、返回最终的FM模型的参数
    return w0, w, v

def getCost(predict, classLabels):
    '''计算预测准确性
    input:  predict(list)预测值
            classLabels(list)标签
    output: error(float)计算损失函数的值
    '''
    m = len(predict)
    error = 0.0
    for i in xrange(m):
        error -=  np.log(sigmoid(predict[i] * classLabels[i] ))  
    return error

def getPrediction(dataMatrix, w0, w, v):
    '''得到预测值
    input:  dataMatrix(mat)特征
            w(int)常数项权重
            w0(int)一次项权重
            v(float)交叉项权重
    output: result(list)预测的结果
    '''
    m = np.shape(dataMatrix)[0]   
    result = []
    for x in xrange(m):
        
        inter_1 = dataMatrix[x] * v
        inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * \
         np.multiply(v, v)  # multiply对应元素相乘
        # 完成交叉项
        interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
        p = w0 + dataMatrix[x] * w + interaction  # 计算预测的输出        
        pre = sigmoid(p[0, 0])        
        result.append(pre)        
    return result

def getAccuracy(predict, classLabels):
    '''计算预测准确性
    input:  predict(list)预测值
            classLabels(list)标签
    output: float(error) / allItem(float)错误率
    '''
    m = len(predict)
    allItem = 0
    error = 0
    for i in xrange(m):
        allItem += 1
        if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
            error += 1
        elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
            error += 1
        else:
            continue
    return float(error) / allItem      

def save_model(file_name, w0, w, v):
    '''保存训练好的FM模型
    input:  file_name(string):保存的文件名
            w0(float):偏置项
            w(mat):一次项的权重
            v(mat):交叉项的权重
    '''
    f = open(file_name, "w")
    # 1、保存w0
    f.write(str(w0) + "\n")
    # 2、保存一次项的权重
    w_array = []
    m = np.shape(w)[0]
    for i in xrange(m):
        w_array.append(str(w[i, 0]))
    f.write("\t".join(w_array) + "\n")
    # 3、保存交叉项的权重
    m1 , n1 = np.shape(v)
    for i in xrange(m1):
        v_tmp = []
        for j in xrange(n1):
            v_tmp.append(str(v[i, j]))
        f.write("\t".join(v_tmp) + "\n")
    f.close()
     
   
if __name__ == "__main__":
    # 1、导入训练数据
    print "---------- 1.load data ---------"
    dataTrain, labelTrain = loadDataSet("data_1.txt")
    print "---------- 2.learning ---------"
    # 2、利用随机梯度训练FM模型
    w0, w, v = stocGradAscent(np.mat(dataTrain), labelTrain, 3, 10000, 0.01)
    predict_result = getPrediction(np.mat(dataTrain), w0, w, v)  # 得到训练的准确性
    print "----------training accuracy: %f" % (1 - getAccuracy(predict_result, labelTrain))
    print "---------- 3.save result ---------"
    # 3、保存训练好的FM模型
    save_model("weights", w0, w, v)
