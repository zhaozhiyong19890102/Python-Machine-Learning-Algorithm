# coding:UTF-8
'''
Date:20160805
@author: zhaozhiyong
'''
import numpy as np

def load_data(inputfile):
    '''导入训练数据
    input:  inputfile(string)训练样本的位置
    output: feature_data(mat)特征
            label_data(mat)标签
            k(int)类别的个数
    '''
    f = open(inputfile)  # 打开文件
    feature_data = []
    label_data = []
    for line in f.readlines():
        feature_tmp = []
        feature_tmp.append(1)  # 偏置项
        lines = line.strip().split("\t")
        for i in xrange(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label_data.append(int(lines[-1]))
        
        feature_data.append(feature_tmp)
    f.close()  # 关闭文件
    return np.mat(feature_data), np.mat(label_data).T, len(set(label_data))

def cost(err, label_data):
    '''计算损失函数值
    input:  err(mat):exp的值
            label_data(mat):标签的值
    output: sum_cost / m(float):损失函数的值
    '''
    m = np.shape(err)[0]
    sum_cost = 0.0
    for i in xrange(m):
        if err[i, label_data[i, 0]] / np.sum(err[i, :]) > 0:
            sum_cost -= np.log(err[i, label_data[i, 0]] / np.sum(err[i, :]))
        else:
            sum_cost -= 0
    return sum_cost / m
    

def gradientAscent(feature_data, label_data, k, maxCycle, alpha):
    '''利用梯度下降法训练Softmax模型
    input:  feature_data(mat):特征
            label_data(mat):标签
            k(int):类别的个数
            maxCycle(int):最大的迭代次数
            alpha(float):学习率
    output: weights(mat)：权重
    '''
    m, n = np.shape(feature_data)
    weights = np.mat(np.ones((n, k)))  # 权重的初始化
    i = 0
    while i <= maxCycle:
        err = np.exp(feature_data * weights)
        if i % 500 == 0:
            print "\t-----iter: ", i , ", cost: ", cost(err, label_data)
        rowsum = -err.sum(axis=1)
        rowsum = rowsum.repeat(k, axis=1)
        err = err / rowsum
        for x in range(m):
            err[x, label_data[x, 0]] += 1
        weights = weights + (alpha / m) * feature_data.T * err      
        i += 1           
    return weights

def save_model(file_name, weights):
    '''保存最终的模型
    input:  file_name(string):保存的文件名
            weights(mat):softmax模型
    '''
    f_w = open(file_name, "w")
    m, n = np.shape(weights)
    for i in xrange(m):
        w_tmp = []
        for j in xrange(n):
            w_tmp.append(str(weights[i, j]))
        f_w.write("\t".join(w_tmp) + "\n")
    f_w.close()
            
if __name__ == "__main__":
    inputfile = "SoftInput.txt"
    # 1、导入训练数据
    print "---------- 1.load data ------------"
    feature, label, k = load_data(inputfile)
    # 2、训练Softmax模型
    print "---------- 2.training ------------"
    weights = gradientAscent(feature, label, k, 10000, 0.4)
    # 3、保存最终的模型
    print "---------- 3.save model ------------"
    save_model("weights", weights)
