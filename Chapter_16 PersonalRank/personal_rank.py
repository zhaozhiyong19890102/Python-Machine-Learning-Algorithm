# coding=utf-8
'''
Date:20160930
@author: zhaozhiyong
'''

import numpy as np

def load_data(file_path):
	'''导入用户商品数据
	input:  file_path(string):用户商品数据存储的文件
	output: data(mat):用户商品矩阵
	'''
	f = open(file_path)
	data = []
	for line in f.readlines():
		lines = line.strip().split("\t")
		tmp = []
		for x in lines:
			if x != "-":
				tmp.append(1) # 打过分记为1
			else:
				tmp.append(0) # 未打分记为0
		data.append(tmp)
	f.close()
	return np.mat(data)

def generate_dict(dataTmp):
	'''将用户-商品矩阵转换成二部图的表示
	input:  dataTmp(mat):用户商品矩阵
	output: data_dict(dict):图的表示
	'''
	m, n = np.shape(dataTmp)
	
	data_dict = {}
	# 对每一个用户生成节点
	for i in xrange(m):
		tmp_dict = {}
		for j in xrange(n):
			if dataTmp[i, j] != 0:
				tmp_dict["D_" + str(j)] = dataTmp[i, j]
		data_dict["U_" + str(i)] = tmp_dict
	
	# 对每一个商品生成节点	
	for j in xrange(n):
		tmp_dict = {}
		for i in xrange(m):
			if dataTmp[i, j] != 0:
				tmp_dict["U_" + str(i)] = dataTmp[i, j]
		data_dict["D_" + str(j)] = tmp_dict
	return data_dict
		

def PersonalRank(data_dict, alpha, user, maxCycles):
	'''利用PersonalRank打分
	input:  data_dict(dict):用户-商品的二部图表示
			alpha(float):概率
			user(string):指定用户
			maxCycles(int):最大的迭代次数
	output: rank(dict):打分的列表
	'''
	# 1、初始化打分
	rank = {}  
	for x in data_dict.keys():
		rank[x] = 0
	rank[user] = 1 # 从user开始游走
	
	# 2、迭代
	step = 0
	while step < maxCycles:
		tmp = {}
		for x in data_dict.keys():
			tmp[x] = 0
			
		for i, ri in data_dict.items():
			for j in ri.keys():
				if j not in tmp:
					tmp[j] = 0
				tmp[j] += alpha * rank[i] / (1.0 * len(ri))  
				if j == user:
					tmp[j] += (1 - alpha)
		# 判断是否收敛
		check = []
		for k in tmp.keys():
			check.append(tmp[k] - rank[k])	
		if sum(check) <= 0.0001:
			break
		rank = tmp
		if step % 20 == 0:
			print "iter: ", step
		step = step + 1
	return rank

def recommend(data_dict, rank, user):
	'''得到最终的推荐列表
	input:  data_dict(dict):用户-商品的二部图表示
			rank(dict):打分的结果
			user(string):用户
	output: result(dict):推荐结果
	'''
	items_dict = {}
	# 1、用户user已打过分的项
	items = []
	for k in data_dict[user].keys():
		items.append(k)
		
	# 2、从rank取出商品的打分
	for k in rank.keys():
		if k.startswith("D_"): # 商品
			if k not in items: # 排除已经互动过的商品
				items_dict[k] = rank[k]
			
	# 3、按打分的降序排序
	result = sorted(items_dict.items(), key=lambda d: d[1], reverse=True)
	return result

if __name__ == "__main__":
	# 1、导入用户商品矩阵
	print "------------ 1.load data -------------"
	dataMat = load_data("data.txt")
	# 2、将用户商品矩阵转换成邻接表的存储
	print "------------ 2.generate dict --------------"
	data_dict = generate_dict(dataMat)
	# 3、利用PersonalRank计算
	print "------------ 3.PersonalRank --------------"
	rank = PersonalRank(data_dict, 0.85, "U_0", 500)
	# 4、根据rank结果进行商品推荐
	print "------------ 4.recommend -------------"
	result = recommend(data_dict, rank, "U_0")
	print result
