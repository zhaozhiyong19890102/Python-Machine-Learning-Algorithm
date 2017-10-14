# coding:UTF-8
'''
Date:20161030
@author: zhaozhiyong
'''

import numpy as np
import random as rd
from math import log 
from tree import build_tree, predict
import cPickle as pickle

def load_data(file_name):
    '''导入数据
    input:  file_name(string):训练数据保存的文件名
    output: data_train(list):训练数据
    '''
    data_train = []
    f = open(file_name)
    for line in f.readlines():
        lines = line.strip().split("\t")
        data_tmp = []
        for x in lines:
            data_tmp.append(float(x))
        data_train.append(data_tmp)
    f.close()
    return data_train

def choose_samples(data, k):
    '''
    input:  data(list):原始数据集
            k(int):选择特征的个数
    output: data_samples(list):被选择出来的样本
            feature(list):被选择的特征index
    '''
    m, n = np.shape(data)  # 样本的个数和样本特征的个数
    # 1、选择出k个特征的index
    feature = []
    for j in xrange(k):
        feature.append(rd.randint(0, n - 2))  # n-1列是标签
    # 2、选择出m个样本的index
    index = []
    for i in xrange(m):
        index.append(rd.randint(0, m - 1))
    # 3、从data中选择出m个样本的k个特征，组成数据集data_samples
    data_samples = []
    for i in xrange(m):
        data_tmp = []
        for fea in feature:
            data_tmp.append(data[index[i]][fea])
        data_tmp.append(data[index[i]][-1])
        data_samples.append(data_tmp)
    return data_samples, feature


def random_forest_training(data_train, trees_num):
    '''构建随机森林
    input:  data_train(list):训练数据
            trees_num(int):分类树的个数
    output: trees_result(list):每一棵树的最好划分
            trees_feature(list):每一棵树中对原始特征的选择
    '''
    trees_result = []  # 构建好每一棵树的最好划分
    trees_feature = []
    n = np.shape(data_train)[1]  # 样本的维数
    if n > 2:
        k = int(log(n - 1, 2)) + 1 # 设置特征的个数
    else:
        k = 1
    # 开始构建每一棵树
    for i in xrange(trees_num):
        # 1、随机选择m个样本, k个特征
        data_samples, feature = choose_samples(data_train, k)
        # 2、构建每一棵分类树
        tree = build_tree(data_samples)
        # 3、保存训练好的分类树
        trees_result.append(tree)
        # 4、保存好该分类树使用到的特征
        trees_feature.append(feature)
    
    return trees_result, trees_feature

def split_data(data_train, feature):
    '''选择特征
    input:  data_train(list):训练数据集
            feature(list):要选择的特征
    output: data(list):选择出来的数据集
    '''
    m = np.shape(data_train)[0]
    data = []
    
    for i in xrange(m):
        data_x_tmp = []
        for x in feature:
            data_x_tmp.append(data_train[i][x])
        data_x_tmp.append(data_train[i][-1])
        data.append(data_x_tmp)
    return data
            

def get_predict(trees_result, trees_fiture, data_train):
    m_tree = len(trees_result)
    m = np.shape(data_train)[0]
    
    result = []
    for i in xrange(m_tree):
        clf = trees_result[i]
        feature = trees_fiture[i]
        data = split_data(data_train, feature)
        result_i = []
        for i in xrange(m):
            result_i.append((predict(data[i][0:-1], clf).keys())[0])
        result.append(result_i)
    final_predict = np.sum(result, axis=0)
    return final_predict

def cal_correct_rate(data_train, final_predict):
    m = len(final_predict)
    corr = 0.0
    for i in xrange(m):
        if data_train[i][-1] * final_predict[i] > 0:
            corr += 1
    return corr / m

def save_model(trees_result, trees_feature, result_file, feature_file):
    # 1、保存选择的特征
    m = len(trees_feature)
    f_fea = open(feature_file, "w")
    for i in xrange(m):
        fea_tmp = []
        for x in trees_feature[i]:
            fea_tmp.append(str(x))
        f_fea.writelines("\t".join(fea_tmp) + "\n")
    f_fea.close()
    
    # 2、保存最终的随机森林模型
    with open(result_file, 'w') as f:
        pickle.dump(trees_result, f)
        

if __name__ == "__main__":
    # 1、导入数据
    print "----------- 1、load data -----------"
    data_train = load_data("data.txt")
    # 2、训练random_forest模型
    print "----------- 2、random forest training ------------"
    trees_result, trees_feature = random_forest_training(data_train, 50)
    # 3、得到训练的准确性
    print "------------ 3、get prediction correct rate ------------"
    result = get_predict(trees_result, trees_feature, data_train)
    corr_rate = cal_correct_rate(data_train, result)
    print "\t------correct rate: ", corr_rate
    # 4、保存最终的随机森林模型
    print "------------ 4、save model -------------"
    save_model(trees_result, trees_feature, "result_file", "feature_file")
