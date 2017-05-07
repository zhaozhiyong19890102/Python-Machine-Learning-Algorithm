# coding:UTF-8
'''
Date:20160923
@author: zhaozhiyong
'''

import numpy as np
import math

MinPts = 5  # 定义半径内的最少的数据点的个数

def load_data(file_path):
    '''导入数据
    input:  file_path(string):文件名
    output: data(mat):数据
    '''
    f = open(file_path)
    data = []
    for line in f.readlines():
        data_tmp = []
        lines = line.strip().split("\t")
        for x in lines:
            data_tmp.append(float(x.strip()))
        data.append(data_tmp)
    f.close()
    return np.mat(data)

def epsilon(data, MinPts):
    '''计算半径
    input:  data(mat):训练数据
            MinPts(int):半径内的数据点的个数
    output: eps(float):半径
    '''
    m, n = np.shape(data)
    xMax = np.max(data, 0)
    xMin = np.min(data, 0)
    eps = ((np.prod(xMax - xMin) * MinPts * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
    return eps
    
def distance(data):
    m, n = np.shape(data)
    dis = np.mat(np.zeros((m, m)))
    for i in xrange(m):
        for j in xrange(i, m):
            # 计算i和j之间的欧式距离
            tmp = 0
            for k in xrange(n):
                tmp += (data[i, k] - data[j, k]) * (data[i, k] - data[j, k])
            dis[i, j] = np.sqrt(tmp)
            dis[j, i] = dis[i, j]
    return dis

def find_eps(distance_D, eps):
    ind = []
    n = np.shape(distance_D)[1]
    for j in xrange(n):
        if distance_D[0, j] <= eps:
            ind.append(j)
    return ind

def dbscan(data, eps, MinPts):
    m = np.shape(data)[0]
    # 区分核心点1，边界点0和噪音点-1
    types = np.mat(np.zeros((1, m)))
    sub_class = np.mat(np.zeros((1, m)))
    # 用于判断该点是否处理过，0表示未处理过
    dealed = np.mat(np.zeros((m, 1)))
    # 计算每个数据点之间的距离
    dis = distance(data)
    # 用于标记类别
    number = 1
    
    # 对每一个点进行处理
    for i in xrange(m):
        # 找到未处理的点
        if dealed[i, 0] == 0:
            # 找到第i个点到其他所有点的距离
            D = dis[i, ]
            # 找到半径eps内的所有点
            ind = find_eps(D, eps)
            # 区分点的类型
            # 边界点
            if len(ind) > 1 and len(ind) < MinPts + 1:
                types[0, i] = 0
                sub_class[0, i] = 0
            # 噪音点
            if len(ind) == 1:
                types[0, i] = -1
                sub_class[0, i] = -1
                dealed[i, 0] = 1
            # 核心点
            if len(ind) >= MinPts + 1:
                types[0, i] = 1
                for x in ind:
                    sub_class[0, x] = number
                # 判断核心点是否密度可达
                while len(ind) > 0:
                    dealed[ind[0], 0] = 1
                    D = dis[ind[0], ]
                    tmp = ind[0]
                    del ind[0]
                    ind_1 = find_eps(D, eps)
                    
                    if len(ind_1) > 1:  # 处理非噪音点
                        for x1 in ind_1:
                            sub_class[0, x1] = number
                        if len(ind_1) >= MinPts + 1:
                            types[0, tmp] = 1
                        else:
                            types[0, tmp] = 0
                            
                        for j in xrange(len(ind_1)):
                            if dealed[ind_1[j], 0] == 0:
                                dealed[ind_1[j], 0] = 1
                                ind.append(ind_1[j])
                                sub_class[0, ind_1[j]] = number
                number += 1
    
    # 最后处理所有未分类的点为噪音点
    ind_2 = ((sub_class == 0).nonzero())[1]
    for x in ind_2:
        sub_class[0, x] = -1
        types[0, x] = -1
        
    return types, sub_class

def save_result(file_name, source):
    f = open(file_name, "w")
    n = np.shape(source)[1]
    tmp = []
    for i in xrange(n):
        tmp.append(str(source[0, i]))
    f.write("\n".join(tmp))
    f.close()    

if __name__ == "__main__":
    # 1、导入数据
    print "----------- 1、load data ----------"
    data = load_data("data.txt")
    # 2、计算半径
    print "----------- 2、calculate eps ----------"
    eps = epsilon(data, MinPts)
    # 3、利用DBSCAN算法进行训练
    print "----------- 3、DBSCAN -----------"
    types, sub_class = dbscan(data, eps, MinPts)
    # 4、保存最终的结果
    print "----------- 4、save result -----------"
    save_result("types", types)
    save_result("sub_class", sub_class)
