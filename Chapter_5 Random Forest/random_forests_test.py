#coding:UTF-8
'''
Date:20161030
@author: zhaozhiyong
'''

import cPickle as pickle
from random_forests_train import get_predict

def load_data(file_name):
    '''导入待分类的数据集
    input:  file_name(string):待分类数据存储的位置
    output: test_data(list)
    '''
    f = open(file_name)
    test_data = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        tmp = []
        for x in lines:
            tmp.append(float(x))
        tmp.append(0) # 保存初始的label
        test_data.append(tmp)
    f.close()
    return test_data

def load_model(result_file, feature_file):
    '''导入随机森林模型和每一个分类树中选择的特征
    input:  result_file(string):随机森林模型存储的文件
            feature_file(string):分类树选择的特征存储的文件
    output: trees_result(list):随机森林模型
            trees_fiture(list):每一棵分类树选择的特征
    '''
    # 1、导入选择的特征
    trees_fiture = []
    f_fea = open(feature_file)
    for line in f_fea.readlines():
        lines = line.strip().split("\t")
        tmp = []
        for x in lines:
            tmp.append(int(x))
        trees_fiture.append(tmp)
    f_fea.close()
    
    # 2、导入随机森林模型
    with open(result_file, 'r') as f:
        trees_result = pickle.load(f)
    
    return trees_result, trees_fiture
    
def save_result(data_test, prediction, result_file):
    '''保存最终的预测结果
    input:  data_test(list):待预测的数据
            prediction(list):预测的结果
            result_file(string):存储最终预测结果的文件名
    '''
    m = len(prediction)
    n = len(data_test[0])
    
    f_result = open(result_file, "w")
    for i in xrange(m):
        tmp = []
        for j in xrange(n -1):
            tmp.append(str(data_test[i][j]))
        tmp.append(str(prediction[i]))
        f_result.writelines("\t".join(tmp) + "\n")
    f_result.close()

if __name__ == "__main__":
    # 1、导入测试数据集
    print "--------- 1、load test data --------"
    data_test = load_data("test_data.txt")
    # 2、导入随机森林模型
    print "--------- 2、load random forest model ----------"
    trees_result, trees_feature = load_model("result_file", "feature_file")
    # 3、预测
    print "--------- 3、get prediction -----------"
    prediction = get_predict(trees_result, trees_feature, data_test)
    # 4、保存最终的预测结果
    print "--------- 4、save result -----------"
    save_result(data_test, prediction, "final_result")
    
    
    
