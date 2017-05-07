# coding:UTF-8
'''
Date:20160426
@author: zhaozhiyong
'''
import math
import numpy as np

MIN_DISTANCE = 0.000001  # mini error

def load_data(path, feature_num=2):
    '''导入数据
    input:  path(string)文件的存储位置
            feature_num(int)特征的个数
    output: data(array)特征
    '''
    f = open(path)  # 打开文件
    data = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        data_tmp = []
        if len(lines) != feature_num:  # 判断特征的个数是否正确
            continue
        for i in xrange(feature_num):
            data_tmp.append(float(lines[i]))
        data.append(data_tmp)
    f.close()  # 关闭文件
    return data

def gaussian_kernel(distance, bandwidth):
    '''高斯核函数
    input:  distance(mat):欧式距离
            bandwidth(int):核函数的带宽
    output: gaussian_val(mat):高斯函数值
    '''
    m = np.shape(distance)[0]  # 样本个数
    right = np.mat(np.zeros((m, 1)))  # mX1的矩阵
    for i in xrange(m):
        right[i, 0] = (-0.5 * distance[i] * distance[i].T) / (bandwidth * bandwidth)
        right[i, 0] = np.exp(right[i, 0])
    left = 1 / (bandwidth * math.sqrt(2 * math.pi))
    
    gaussian_val = left * right
    return gaussian_val

def shift_point(point, points, kernel_bandwidth):
    '''计算均值漂移点
    input:  point(mat)需要计算的点
            points(array)所有的样本点
            kernel_bandwidth(int)核函数的带宽
    output: point_shifted(mat)漂移后的点
    '''
    points = np.mat(points)
    m = np.shape(points)[0]  # 样本的个数
    # 计算距离
    point_distances = np.mat(np.zeros((m, 1)))
    for i in xrange(m):
        point_distances[i, 0] = euclidean_dist(point, points[i])
    
    # 计算高斯核        
    point_weights = gaussian_kernel(point_distances, kernel_bandwidth)  # mX1的矩阵
    
    # 计算分母
    all_sum = 0.0
    for i in xrange(m):
        all_sum += point_weights[i, 0]
    
    # 均值偏移
    point_shifted = point_weights.T * points / all_sum
    return point_shifted

def euclidean_dist(pointA, pointB):
    '''计算欧式距离
    input:  pointA(mat):A点的坐标
            pointB(mat):B点的坐标
    output: math.sqrt(total):两点之间的欧式距离
    '''
    # 计算pointA和pointB之间的欧式距离
    total = (pointA - pointB) * (pointA - pointB).T
    return math.sqrt(total)  # 欧式距离

def group_points(mean_shift_points):
    '''计算所属的类别
    input:  mean_shift_points(mat):漂移向量
    output: group_assignment(array):所属类别
    '''
    group_assignment = []
    m, n = np.shape(mean_shift_points)
    index = 0
    index_dict = {}
    for i in xrange(m):
        item = []
        for j in xrange(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))
        
        item_1 = "_".join(item)
        if item_1 not in index_dict:
            index_dict[item_1] = index
            index += 1
    
    for i in xrange(m):
        item = []
        for j in xrange(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))

        item_1 = "_".join(item)
        group_assignment.append(index_dict[item_1])

    return group_assignment

def train_mean_shift(points, kenel_bandwidth=2):
    '''训练Mean shift模型
    input:  points(array):特征数据
            kenel_bandwidth(int):核函数的带宽
    output: points(mat):特征点
            mean_shift_points(mat):均值漂移点
            group(array):类别
    '''
    mean_shift_points = np.mat(points)
    max_min_dist = 1
    iteration = 0  # 训练的代数
    m = np.shape(mean_shift_points)[0]  # 样本的个数
    need_shift = [True] * m  # 标记是否需要漂移

    # 计算均值漂移向量
    while max_min_dist > MIN_DISTANCE:
        max_min_dist = 0
        iteration += 1
        print "\titeration : " + str(iteration)
        for i in range(0, m):
            # 判断每一个样本点是否需要计算偏移均值
            if not need_shift[i]:
                continue
            p_new = mean_shift_points[i]
            p_new_start = p_new
            p_new = shift_point(p_new, points, kenel_bandwidth)  # 对样本点进行漂移
            dist = euclidean_dist(p_new, p_new_start)  # 计算该点与漂移后的点之间的距离

            if dist > max_min_dist:
                max_min_dist = dist
            if dist < MIN_DISTANCE:  # 不需要移动
                need_shift[i] = False

            mean_shift_points[i] = p_new

    # 计算最终的group
    group = group_points(mean_shift_points)  # 计算所属的类别
    
    return np.mat(points), mean_shift_points, group

def save_result(file_name, data):
    '''保存最终的计算结果
    input:  file_name(string):存储的文件名
            data(mat):需要保存的文件
    '''
    f = open(file_name, "w")
    m, n = np.shape(data)
    for i in xrange(m):
        tmp = []
        for j in xrange(n):
            tmp.append(str(data[i, j]))
        f.write("\t".join(tmp) + "\n")
    f.close()
    

if __name__ == "__main__":
    # 导入数据集
    print "----------1.load data ------------"
    data = load_data("data", 2)
    # 训练，h=2
    print "----------2.training ------------"
    points, shift_points, cluster = train_mean_shift(data, 2)
    # 保存所属的类别文件
    print "----------3.1.save sub ------------"
    save_result("sub_1", np.mat(cluster))
    print "----------3.2.save center ------------"
    # 保存聚类中心
    save_result("center_1", shift_points)    

