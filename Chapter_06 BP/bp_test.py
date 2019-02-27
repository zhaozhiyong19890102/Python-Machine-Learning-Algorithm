# coding:UTF-8
'''
Date:20160831
@author: zhaozhiyong
'''
import numpy as np
from bp_train import get_predict

def load_data(file_name):
    '''导入数据
    input:  file_name(string):文件的存储位置
    output: feature_data(mat):特征
    '''
    f = open(file_name)  # 打开文件
    feature_data = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        for i in xrange(len(lines)):
            feature_tmp.append(float(lines[i]))        
        feature_data.append(feature_tmp)
    f.close()  # 关闭文件
    return np.mat(feature_data)

def generate_data():
    '''在[-4.5,4.5]之间随机生成20000组点
    '''
    # 1、随机生成数据点
    data = np.mat(np.zeros((20000, 2)))
    m = np.shape(data)[0]
    x = np.mat(np.random.rand(20000, 2))
    for i in xrange(m):
        data[i, 0] = x[i, 0] * 9 - 4.5
        data[i, 1] = x[i, 1] * 9 - 4.5
    # 2、将数据点保存到文件“test_data”中
    f = open("test_data", "w")
    m,n = np.shape(data)
    for i in xrange(m):
        tmp =[]
        for j in xrange(n):
            tmp.append(str(data[i,j]))
        f.write("\t".join(tmp) + "\n")
    f.close()       

def load_model(file_w0, file_w1, file_b0, file_b1):
    
    def get_model(file_name):
        f = open(file_name)
        model = []
        for line in f.readlines():
            lines = line.strip().split("\t")
            model_tmp = []
            for x in lines:
                model_tmp.append(float(x.strip()))
            model.append(model_tmp)
        f.close()
        return np.mat(model)
    
    # 1、导入输入层到隐含层之间的权重
    w0 = get_model(file_w0)
    
    # 2、导入隐含层到输出层之间的权重
    w1 = get_model(file_w1)
    
    # 3、导入输入层到隐含层之间的权重
    b0 = get_model(file_b0)
    
    # 4、导入隐含层到输出层之间的权重
    b1 = get_model(file_b1)

    return w0, w1, b0, b1

def save_predict(file_name, pre):
    '''保存最终的预测结果
    input:  pre(mat):最终的预测结果
    output:
    '''
    f = open(file_name, "w")
    m = np.shape(pre)[0]
    result = []
    for i in xrange(m):
        result.append(str(pre[i, 0]))
    f.write("\n".join(result))
    f.close()

if __name__ == "__main__":
    generate_data()
    # 1、导入测试数据
    print "--------- 1.load data ------------"
    dataTest = load_data("test_data")
    # 2、导入BP神经网络模型
    print "--------- 2.load model ------------"
    w0, w1, b0, b1 = load_model("weight_w0", "weight_w1", "weight_b0", "weight_b1")
    # 3、得到最终的预测值
    print "--------- 3.get prediction ------------"
    result = get_predict(dataTest, w0, w1, b0, b1)
    # 4、保存最终的预测结果
    print "--------- 4.save result ------------"
    pre = np.argmax(result, axis=1)
    save_predict("result", pre)
