# coding:UTF-8
'''
Date:20160928
@author: zhaozhiyong
'''

import numpy as np

def load_data(file_path):
    '''导入训练数据
    input:  file_path(string):训练数据
    output: feature(mat):特征
            label(mat):标签
    '''
    f = open(file_path)
    feature = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        feature_tmp.append(1)  # x0
        for i in xrange(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
        label.append(float(lines[-1]))
    f.close()
    return np.mat(feature), np.mat(label).T

def ridge_regression(feature, label, lam):
    '''最小二乘的求解方法
    input:  feature(mat):特征
            label(mat):标签
    output: w(mat):回归系数
    '''
    n = np.shape(feature)[1]
    w = (feature.T * feature + lam * np.mat(np.eye(n))).I * feature.T * label
    return w

def get_gradient(feature, label, w, lam):
    '''计算导函数的值
    input:  feature(mat):特征
            label(mat):标签
    output: w(mat):回归系数
    '''
    err = (label - feature * w).T   
    left = err * (-1) * feature 
    return left.T + lam * w

def get_result(feature, label, w, lam):
    '''
    input:  feature(mat):特征
            label(mat):标签
    output: w(mat):回归系数
    '''
    left = (label - feature * w).T * (label - feature * w)
    right = lam * w.T * w
    return (left + right) / 2

def get_error(feature, label, w):
    '''
    input:  feature(mat):特征
            label(mat):标签
    output: w(mat):回归系数
    '''
    m = np.shape(feature)[0]
    left = (label - feature * w).T * (label - feature * w)
    return (left / (2 * m))[0, 0]
    

def bfgs(feature, label, lam, maxCycle):
    '''利用bfgs训练Ridge Regression模型
    input:  feature(mat):特征
            label(mat):标签
            lam(float):正则化参数
            maxCycle(int):最大迭代次数
    output: w(mat):回归系数
    '''
    n = np.shape(feature)[1]
    # 1、初始化
    w0 = np.mat(np.zeros((n, 1)))  
    rho = 0.55  
    sigma = 0.4  
    Bk = np.eye(n)  
    k = 1  
    while (k < maxCycle):
        print "\titer: ", k, "\terror: ", get_error(feature, label, w0) 
        gk = get_gradient(feature, label, w0, lam)  # 计算梯度  
        dk = np.mat(-np.linalg.solve(Bk, gk))  
        m = 0  
        mk = 0  
        while (m < 20):  
            newf = get_result(feature, label, (w0 + rho ** m * dk), lam)  
            oldf = get_result(feature, label, w0, lam)  
            if (newf < oldf + sigma * (rho ** m) * (gk.T * dk)[0, 0]):  
                mk = m  
                break  
            m = m + 1  
          
        # BFGS校正  
        w = w0 + rho ** mk * dk  
        sk = w - w0  
        yk = get_gradient(feature, label, w, lam) - gk 
        if (yk.T * sk > 0):  
            Bk = Bk - (Bk * sk * sk.T * Bk) / (sk.T * Bk * sk) + (yk * yk.T) / (yk.T * sk)  
          
        k = k + 1  
        w0 = w  
    return w0 

def lbfgs(feature, label, lam, maxCycle, m=10):
    '''利用lbfgs训练Ridge Regression模型
    input:  feature(mat):特征
            label(mat):标签
            lam(float):正则化参数
            maxCycle(int):最大迭代次数
            m(int):lbfgs中选择保留的个数
    output: w(mat):回归系数
    '''
    n = np.shape(feature)[1]
    # 1、初始化
    w0 = np.mat(np.zeros((n, 1)))
    rho = 0.55
    sigma = 0.4
    
    H0 = np.eye(n)
    
    s = []
    y = []
    
    k = 1
    gk = get_gradient(feature, label, w0, lam)  # 3X1
    print gk
    dk = -H0 * gk
    # 2、迭代
    while (k < maxCycle):
        print "iter: ", k, "\terror: ", get_error(feature, label, w0) 
        m = 0
        mk = 0
        gk = get_gradient(feature, label, w0, lam)
        # 2.1、Armijo线搜索
        while (m < 20):
            newf = get_result(feature, label, (w0 + rho ** m * dk), lam)
            oldf = get_result(feature, label, w0, lam)
            if newf < oldf + sigma * (rho ** m) * (gk.T * dk)[0, 0]:
                mk = m
                break
            m = m + 1
        
        # 2.2、LBFGS校正
        w = w0 + rho ** mk * dk
        
        # 保留m个
        if k > m:
            s.pop(0)
            y.pop(0)
        
        # 保留最新的
        sk = w - w0
        qk = get_gradient(feature, label, w, lam)  # 3X1
        yk = qk - gk
        
        s.append(sk)
        y.append(yk)
        
        # two-loop
        t = len(s)
        a = []
        for i in xrange(t):
            alpha = (s[t - i - 1].T * qk) / (y[t - i - 1].T * s[t - i - 1])
            qk = qk - alpha[0, 0] * y[t - i - 1]
            a.append(alpha[0, 0])
        r = H0 * qk
        
        for i in xrange(t):
            beta = (y[i].T * r) / (y[i].T * s[i])
            r = r + s[i] * (a[t - i - 1] - beta[0, 0])
            
        if yk.T * sk > 0:
            print "update OK!!!!"
            dk = -r
        
        k = k + 1
        w0 = w
    return w0

def save_weights(file_name, w0):
    '''保存最终的结果
    input:  file_name(string):需要保存的文件
            w0(mat):权重
    '''
    f_result = open("weights", "w")
    m, n = np.shape(w0)
    for i in xrange(m):
        w_tmp = []
        for j in xrange(n):
            w_tmp.append(str(w0[i, j]))
        f_result.write("\t".join(w_tmp) + "\n")
    f_result.close()


if __name__ == "__main__":
    # 1、导入数据
    print "----------1.load data ------------"
    feature, label = load_data("data.txt")
    # 2、训练模型
    print "----------2.training ridge_regression ------------"
    method = "lbfgs"  # 选择的方法
    if method == "bfgs":  # 选择BFGS训练模型
        w0 = bfgs(feature, label, 0.5, 1000)
    elif method == "lbfgs":  # 选择L-BFGS训练模型
        w0 = lbfgs(feature, label, 0.5, 1000, m=10)
    else:  # 使用最小二乘的方法
        w0 = ridge_regression(feature, label, 0.5)
    # 3、保存最终的模型
    print "----------3.save model ------------"
    save_weights("weights", w0)
