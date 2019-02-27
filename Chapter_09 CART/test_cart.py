# coding:UTF-8
'''
Date:20161030
@author: zhaozhiyong
'''

import random as rd
import cPickle as pickle
from train_cart import predict,node

def load_data():
    '''导入测试数据集
    '''
    data_test = []
    for i in xrange(400):
        tmp = []
        tmp.append(rd.random())  # 随机生成[0,1]之间的样本
        data_test.append(tmp)
    return data_test

def load_model(tree_file):
    '''导入训练好的CART回归树模型
    input:  tree_file(list):保存CART回归树模型的文件
    output: regression_tree:CART回归树
    '''
    with open(tree_file, 'r') as f:
        regression_tree = pickle.load(f)
    return regression_tree    

def get_prediction(data_test, regression_tree):
    '''对测试样本进行预测
    input:  data_test(list):需要预测的样本
            regression_tree(regression_tree):训练好的回归树模型
    output: result(list):
    '''
    result = []
    for x in data_test:
        result.append(predict(x, regression_tree))
    return result

def save_result(data_test, result, prediction_file):
    '''保存最终的预测结果
    input:  data_test(list):需要预测的数据集
            result(list):预测的结果
            prediction_file(string):保存结果的文件
    '''
    f = open(prediction_file, "w")
    for i in xrange(len(result)):
        a = str(data_test[i][0]) + "\t" + str(result[i]) + "\n"
        f.write(a)
    f.close()
                
if __name__ == "__main__":
    # 1、导入待计算的数据
    print "--------- 1、load data ----------"
    data_test = load_data()
    # 2、导入回归树模型
    print "--------- 2、load regression tree ---------"
    regression_tree = load_model("regression_tree")
    # 3、进行预测
    print "--------- 3、get prediction -----------"
    prediction = get_prediction(data_test, regression_tree)
    # 4、保存预测的结果
    print "--------- 4、save result ----------"
    save_result(data_test, prediction, "prediction")
