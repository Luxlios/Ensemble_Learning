# -*- coding = utf-8 -*-
# @Time : 2021/12/17 14:40
# @Author : Luxlios
# @File : bagging.py
# @Software : PyCharm


import pandas as pd
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm as SVM
from sklearn.metrics import accuracy_score

# 训练集测试集分割
def train_test_split(x, y, test_rate):
    num_x = x.shape[0]
    # 打乱
    index = list(range(num_x))
    random.shuffle(index)
    x = x[index]
    y = y[index]
    
    # 分割
    split = round(num_x * test_rate)
    x_test = x[:split, :]
    x_train = x[split:, :]
    y_test = y[:split]
    y_train = y[split:]
    return x_train, x_test, y_train, y_test

# 自助采样法采样
def bootstrap(x, y):
    # x -- input feature
    # y -- input label
    num_x = x.shape[0]
    bootstrap_x = []
    bootstrap_y = []
    # 采样
    for i in range(num_x):
        index = random.randint(0, num_x-1)
        bootstrap_x.append(x[index, :])
        bootstrap_y.append(y[index])
    # 返回采样得到的训练集
    return bootstrap_x, bootstrap_y

def bagging_knn(x_train, y_train, x_test, y_test, k, n):
    # x_train & x_test -- input feature
    # y_train & y_test -- input label
    # k -- k neighbors
    # n -- n classifers
    
    # save every classifer result -- x_test
    num_x = x_test.shape[0]
    class_result = np.zeros([n, num_x]) 
    
    # bagging
    for i in range(n):
        x, y = bootstrap(x_train, y_train)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x, y)
        class_result[i] = knn.predict(x_test)
    
    # vote 投票得到分类结果
    vote_result = np.zeros(num_x)
    class_result = np.int64(class_result)
    for i in range(num_x):
        vote_result[i] = np.argmax(np.bincount(class_result[:, i]))
    vote_result = np.int64(vote_result)
    accuracy = accuracy_score(vote_result, y_test)
    
    # 返回分类结果和准确度
    return vote_result, accuracy

def bagging_svm(x_train, y_train, x_test, y_test, n, kernel):
    # x_train & x_test -- input feature
    # y_train & y_test -- input label
    # n -- n classifers
    # kernal -- svm
    
    # save every classifer result -- x_test
    num_x = x_test.shape[0]
    class_result = np.zeros([n, num_x])
    
    # bagging
    for i in range(n):
        x, y = bootstrap(x_train, y_train)
        svm = SVM.SVC(C=1, kernel=kernel, decision_function_shape='ovr')
        svm.fit(x, y)
        class_result[i] = svm.predict(x_test)

    # vote 投票得到分类结果
    vote_result = np.zeros(num_x)
    class_result = np.int64(class_result)
    for i in range(num_x):
        vote_result[i] = np.argmax(np.bincount(class_result[:, i]))
    vote_result = np.int64(vote_result)
    accuracy = accuracy_score(vote_result, y_test)
    
    # 返回分类结果和准确度
    return vote_result, accuracy

if __name__ == '__main__':
    # Iris数据集读取
    # 150个样本，4维特征，三种类别
    data = pd.read_csv('.\dataset\iris.data', header=None)
    data = np.array(data)
    x = np.float64(data[:, 0:4])
    y = data[:, 4]
    for i in range(len(y)):
        if y[i] == 'Iris-setosa':
            y[i] = 0
        elif y[i] == 'Iris-versicolor':
            y[i] = 1
        else:
            y[i] = 2  
    y = np.int64(y)
    
    # knn
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_rate=0.368)
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(x_train, y_train)
    result1 = knn.predict(x_test)
    accuracy1 = accuracy_score(result1, y_test)
    # bagging(knn)
    vote_result2, accuracy2 = bagging_knn(x_train, y_train, x_test, y_test, k=11, n=10)
    print('knn(k=11)_accuracy:%.3f'%accuracy1)
    print('bagging_10个knn(k=11)_accuracy:%.3f'%accuracy2)
    
     # svm
    kernel=['linear', 'poly', 'rbf', 'sigmoid']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_rate=0.368)
    svm = SVM.SVC(C=1, kernel=kernel[0], decision_function_shape='ovr')
    svm.fit(x_train, y_train)
    result3 = svm.predict(x_test)
    accuracy3 = accuracy_score(result3, y_test)
    # bagging(svm)
    vote_result4, accuracy4 = bagging_svm(x_train, y_train, x_test, y_test, n=10, kernel=kernel[0])
    print('svm_accuracy:%.3f'%accuracy3)
    print('bagging_10个svm_accuracy:%.3f'%accuracy4)





