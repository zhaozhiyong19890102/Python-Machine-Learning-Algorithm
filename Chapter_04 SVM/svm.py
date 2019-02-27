#coding:UTF-8
import numpy as np
import cPickle as pickle

class SVM:
    def __init__(self, dataSet, labels, C, toler, kernel_option):
        self.train_x = dataSet # 训练特征
        self.train_y = labels  # 训练标签
        self.C = C # 惩罚参数
        self.toler = toler     # 迭代的终止条件之一
        self.n_samples = np.shape(dataSet)[0] # 训练样本的个数
        self.alphas = np.mat(np.zeros((self.n_samples, 1))) # 拉格朗日乘子
        self.b = 0
        self.error_tmp = np.mat(np.zeros((self.n_samples, 2))) # 保存E的缓存
        self.kernel_opt = kernel_option # 选用的核函数及其参数
        self.kernel_mat = calc_kernel(self.train_x, self.kernel_opt) # 核函数的输出

def cal_kernel_value(train_x, train_x_i, kernel_option):
    '''样本之间的核函数的值
    input:  train_x(mat):训练样本
            train_x_i(mat):第i个训练样本
            kernel_option(tuple):核函数的类型以及参数
    output: kernel_value(mat):样本之间的核函数的值
            
    '''
    kernel_type = kernel_option[0] # 核函数的类型，分为rbf和其他
    m = np.shape(train_x)[0] # 样本的个数
    
    kernel_value = np.mat(np.zeros((m, 1)))
    
    if kernel_type == 'rbf': # rbf核函数
        sigma = kernel_option[1]
        if sigma == 0:
            sigma = 1.0
        for i in xrange(m):
            diff = train_x[i, :] - train_x_i
            kernel_value[i] = np.exp(diff * diff.T / (-2.0 * sigma**2))
    else: # 不使用核函数
        kernel_value = train_x * train_x_i.T
    return kernel_value


def calc_kernel(train_x, kernel_option):
    '''计算核函数矩阵
    input:  train_x(mat):训练样本的特征值
            kernel_option(tuple):核函数的类型以及参数
    output: kernel_matrix(mat):样本的核函数的值
    '''
    m = np.shape(train_x)[0] # 样本的个数
    kernel_matrix = np.mat(np.zeros((m, m))) # 初始化样本之间的核函数值
    for i in xrange(m):
        kernel_matrix[:, i] = cal_kernel_value(train_x, train_x[i, :], kernel_option)
    return kernel_matrix

def cal_error(svm, alpha_k):
    '''误差值的计算
    input:  svm:SVM模型
            alpha_k(int):选择出的变量
    output: error_k(float):误差值
    '''
    output_k = float(np.multiply(svm.alphas, svm.train_y).T * svm.kernel_mat[:, alpha_k] + svm.b)
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k


def update_error_tmp(svm, alpha_k):
    '''重新计算误差值
    input:  svm:SVM模型
            alpha_k(int):选择出的变量
    output: 对应误差值
    '''
    error = cal_error(svm, alpha_k)
    svm.error_tmp[alpha_k] = [1, error]

def select_second_sample_j(svm, alpha_i, error_i):
    '''选择第二个样本
    input:  svm:SVM模型
            alpha_i(int):选择出的第一个变量
            error_i(float):E_i
    output: alpha_j(int):选择出的第二个变量
            error_j(float):E_j
    '''
    # 标记为已被优化
    svm.error_tmp[alpha_i] = [1, error_i]
    candidateAlphaList = np.nonzero(svm.error_tmp[:, 0].A)[0]
    
    maxStep = 0
    alpha_j = 0
    error_j = 0

    if len(candidateAlphaList) > 1:
        for alpha_k in candidateAlphaList:
            if alpha_k == alpha_i: 
                continue
            error_k = cal_error(svm, alpha_k)
            if abs(error_k - error_i) > maxStep:
                maxStep = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k
    else: # 随机选择          
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(np.random.uniform(0, svm.n_samples))
        error_j = cal_error(svm, alpha_j)
    
    return alpha_j, error_j

def choose_and_update(svm, alpha_i):
    '''判断和选择两个alpha进行更新
    input:  svm:SVM模型
            alpha_i(int):选择出的第一个变量
    '''
    error_i = cal_error(svm, alpha_i) # 计算第一个样本的E_i
    
    # 判断选择出的第一个变量是否违反了KKT条件
    if (svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or\
        (svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):

        # 1、选择第二个变量
        alpha_j, error_j = select_second_sample_j(svm, alpha_i, error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        # 2、计算上下界
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            return 0

        # 3、计算eta
        eta = 2.0 * svm.kernel_mat[alpha_i, alpha_j] - svm.kernel_mat[alpha_i, alpha_i] \
                  - svm.kernel_mat[alpha_j, alpha_j]
        if eta >= 0:
            return 0

        # 4、更新alpha_j
        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta

        # 5、确定最终的alpha_j
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        if svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L

        # 6、判断是否结束      
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            update_error_tmp(svm, alpha_j)
            return 0

        # 7、更新alpha_i
        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] \
                                * (alpha_j_old - svm.alphas[alpha_j])

        # 8、更新b
        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                                                    * svm.kernel_mat[alpha_i, alpha_i] \
                             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
                                                    * svm.kernel_mat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                                                    * svm.kernel_mat[alpha_i, alpha_j] \
                             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
                                                    * svm.kernel_mat[alpha_j, alpha_j]
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        # 9、更新error
        update_error_tmp(svm, alpha_j)
        update_error_tmp(svm, alpha_i)

        return 1
    else:
        return 0

def SVM_training(train_x, train_y, C, toler, max_iter, kernel_option = ('rbf', 0.431029)):
    '''SVM的训练
    input:  train_x(mat):训练数据的特征
            train_y(mat):训练数据的标签
            C(float):惩罚系数
            toler(float):迭代的终止条件之一
            max_iter(int):最大迭代次数
            kerner_option(tuple):核函数的类型及其参数
    output: svm模型
    '''
    # 1、初始化SVM分类器
    svm = SVM(train_x, train_y, C, toler, kernel_option)
    
    # 2、开始训练
    entireSet = True
    alpha_pairs_changed = 0
    iteration = 0
    
    while (iteration < max_iter) and ((alpha_pairs_changed > 0) or entireSet):
        print "\t iterration: ", iteration
        alpha_pairs_changed = 0

        if entireSet:
            # 对所有的样本
            for x in xrange(svm.n_samples):
                alpha_pairs_changed += choose_and_update(svm, x)
            iteration += 1
        else:
            # 非边界样本
            bound_samples = []
            for i in xrange(svm.n_samples):
                if svm.alphas[i,0] > 0 and svm.alphas[i,0] < svm.C:
                    bound_samples.append(i)
            for x in bound_samples:
                alpha_pairs_changed += choose_and_update(svm, x)
            iteration += 1
        
        # 在所有样本和非边界样本之间交替
        if entireSet:
            entireSet = False
        elif alpha_pairs_changed == 0:
            entireSet = True

    return svm

def svm_predict(svm, test_sample_x):
    '''利用SVM模型对每一个样本进行预测
    input:  svm:SVM模型
            test_sample_x(mat):样本
    output: predict(float):对样本的预测
    '''
    # 1、计算核函数矩阵
    kernel_value = cal_kernel_value(svm.train_x, test_sample_x, svm.kernel_opt)
    # 2、计算预测值
    predict = kernel_value.T * np.multiply(svm.train_y, svm.alphas) + svm.b
    return predict

def cal_accuracy(svm, test_x, test_y):
    '''计算预测的准确性
    input:  svm:SVM模型
            test_x(mat):测试的特征
            test_y(mat):测试的标签
    output: accuracy(float):预测的准确性
    '''
    n_samples = np.shape(test_x)[0] # 样本的个数
    correct = 0.0
    for i in xrange(n_samples):
        # 对每一个样本得到预测值
        predict=svm_predict(svm, test_x[i, :])
        # 判断每一个样本的预测值与真实值是否一致
        if np.sign(predict) == np.sign(test_y[i]):
            correct += 1
    accuracy = correct / n_samples
    return accuracy

def save_svm_model(svm_model, model_file):
    '''保存SVM模型
    input:  svm_model:SVM模型
            model_file(string):SVM模型需要保存到的文件
    '''
    with open(model_file, 'w') as f:
        pickle.dump(svm_model, f)
        
