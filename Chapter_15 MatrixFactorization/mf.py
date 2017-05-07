# coding:UTF-8
'''
Date:20160928
@author: zhaozhiyong
'''
import numpy as np

def load_data(path):
    '''导入数据
    input:  path(string):用户商品矩阵存储的位置
    output: data(mat):用户商品矩阵
    '''
    f = open(path)
    data = []
    for line in f.readlines():
        arr = []
        lines = line.strip().split("\t")
        for x in lines:
            if x != "-":
                arr.append(float(x))
            else:
                arr.append(float(0))
        data.append(arr)
    f.close()
    return np.mat(data)

def gradAscent(dataMat, k, alpha, beta, maxCycles):
    '''利用梯度下降法对矩阵进行分解
    input:  dataMat(mat):用户商品矩阵
            k(int):分解矩阵的参数
            alpha(float):学习率
            beta(float):正则化参数
            maxCycles(int):最大迭代次数
    output: p,q(mat):分解后的矩阵
    '''
    m, n = np.shape(dataMat)
    # 1、初始化p和q
    p = np.mat(np.random.random((m, k)))
    q = np.mat(np.random.random((k, n)))
    
    # 2、开始训练
    for step in xrange(maxCycles):
        for i in xrange(m):
            for j in xrange(n):
                if dataMat[i, j] > 0:
                    error = dataMat[i, j]
                    for r in xrange(k):
                        error = error - p[i, r] * q[r, j]
                    for r in xrange(k):
                        # 梯度上升
                        p[i, r] = p[i, r] + alpha * (2 * error * q[r, j] - beta * p[i, r])
                        q[r, j] = q[r, j] + alpha * (2 * error * p[i, r] - beta * q[r, j])

        loss = 0.0
        for i in xrange(m):
            for j in xrange(n):
                if dataMat[i, j] > 0:
                    error = 0.0
                    for r in xrange(k):
                        error = error + p[i, r] * q[r, j]
                    # 3、计算损失函数
                    loss = (dataMat[i, j] - error) * (dataMat[i, j] - error)
                    for r in xrange(k):
                        loss = loss + beta * (p[i, r] * p[i, r] + q[r, j] * q[r, j]) / 2

        if loss < 0.001:
            break
        if step % 1000 == 0:
            print "\titer: ", step, " loss: ", loss
    return p, q

def save_file(file_name, source):
    '''保存结果
    input:  file_name(string):需要保存的文件名
            source(mat):需要保存的文件
    '''
    f = open(file_name, "w")
    m, n = np.shape(source)
    for i in xrange(m):
        tmp = []
        for j in xrange(n):
            tmp.append(str(source[i, j]))
        f.write("\t".join(tmp) + "\n")
    f.close()
    
def prediction(dataMatrix, p, q, user):
    '''为用户user未互动的项打分
    input:  dataMatrix(mat):原始用户商品矩阵
            p(mat):分解后的矩阵p
            q(mat):分解后的矩阵q
            user(int):用户的id
    output: predict(list):推荐列表
    '''
    n = np.shape(dataMatrix)[1]
    predict = {}
    for j in xrange(n):
        if dataMatrix[user, j] == 0:
            predict[j] = (p[user,] * q[:,j])[0,0]
    
    # 按照打分从大到小排序
    return sorted(predict.items(), key=lambda d:d[1], reverse=True)

def top_k(predict, k):
    '''为用户推荐前k个商品
    input:  predict(list):排好序的商品列表
            k(int):推荐的商品个数
    output: top_recom(list):top_k个商品
    '''
    top_recom = []
    len_result = len(predict)
    if k >= len_result:
        top_recom = predict
    else:
        for i in xrange(k):
            top_recom.append(predict[i])
    return top_recom
    


if __name__ == "__main__":
    # 1、导入用户商品矩阵
    print "----------- 1、load data -----------"
    dataMatrix = load_data("data.txt")
    # 2、利用梯度下降法对矩阵进行分解
    print "----------- 2、training -----------"
    p, q = gradAscent(dataMatrix, 5, 0.0002, 0.02, 5000)
    # 3、保存分解后的结果
    print "----------- 3、save decompose -----------"
    save_file("p", p)
    save_file("q", q)
    # 4、预测
    print "----------- 4、prediction -----------"
    predict = prediction(dataMatrix, p, q, 0)
    # 进行Top-K推荐
    print "----------- 5、top_k recommendation ------------"
    top_recom = top_k(predict, 2)
    print top_recom
    print p*q
