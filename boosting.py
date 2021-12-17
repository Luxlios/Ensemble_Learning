# -*- coding = utf-8 -*-
# @Time : 2021/12/17 15:40
# @Author : Luxlios
# @File : 1.py
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

def boosting_knn(x_train, y_train, x_test, y_test, k, m, n):
    # x_train & x_test -- input feature
    # y_train & y_test -- input label
    # k -- k neighbors
    # m -- m classifers
    # n -- n weak classifers

    # save every classifer result -- x_train_test x_test
    num_x1 = x_train.shape[0]
    num_x2 = x_test.shape[0]
    class_result_train = np.zeros([m, num_x1]) 
    class_result_test = np.zeros([m, num_x2])
    
    D = np.zeros([m, num_x1])    # weight of data
    alpha = np.zeros(m)     # weight of classifer
    D[0] = 1/num_x1
    
    # boosting
    for i in range(m):
        error = 1
        for j in range(n):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            class_temp1 = knn.predict(x_train)
            class_temp2 = knn.predict(x_test)
            error_temp = 0
            for l in range(num_x1):
                if class_temp1[l] != y_train[l]:
                    error_temp += D[i, l]
            if error_temp < error:
                error = error_temp
                class_result_train[i] = class_temp1
                class_result_test[i] = class_temp2
                
        # save weight of classifer
        alpha[i] = 0.5*(np.log((1-error)/(error+0.001)))
        # updata weight of data
        for l in range(num_x1):
            if class_result_train[i, l] != y_train[l]:
                D[i, l] = D[i, l]/(2 * (error+0.001))
            else:
                D[i, l] = D[i, l] / (2 * (1-error))
        
    # vote -- 根据权重对标签投票，最多的则为预测结果
    result = np.zeros(num_x2)
    for i in range(num_x2):
        vote_0 = 0
        vote_1 = 0
        vote_2 = 0
        for j in range(m):
            if class_result_test[j, i] == 0:
                vote_0 += alpha[j]
            elif class_result_test[j, i] == 1:
                vote_1 += alpha[j]
            else:
                vote_2 += alpha[j]
        vote = [vote_0, vote_1, vote_2]
        result[i] = vote.index(max(vote))
        
    result = np.int64(result)
    accuracy = accuracy_score(result, y_test)
    
    # 返回分类结果
    return result, accuracy

def boosting_svm(x, y, m, n, kernel):
    # x_train & x_test -- input feature
    # y_train & y_test -- input label
    # k -- k neighbors
    # m -- m classifers
    # n -- n weak classifers

    # save every classifer result -- x_train_test x_test
    num_x1 = x_train.shape[0]
    num_x2 = x_test.shape[0]
    class_result_train = np.zeros([m, num_x1]) 
    class_result_test = np.zeros([m, num_x2])
    
    D = np.zeros([m, num_x1])    # weight of data
    alpha = np.zeros(m)     # weight of classifer
    D[0] = 1/num_x1
    
    # boosting
    for i in range(m):
        error = 1
        for j in range(n):
            svm = SVM.SVC(C=1, kernel=kernel, decision_function_shape='ovr')
            svm.fit(x_train, y_train)
            class_temp1 = svm.predict(x_train)
            class_temp2 = svm.predict(x_test)
            error_temp = 0
            for l in range(num_x1):
                if class_temp1[l] != y_train[l]:
                    error_temp += D[i, l]
            if error_temp < error:
                error = error_temp
                class_result_train[i] = class_temp1
                class_result_test[i] = class_temp2
                
        # save weight of classifer
        alpha[i] = 0.5*(np.log((1-error)/(error+0.001)))
        # updata weight of data
        for l in range(num_x1):
            if class_result_train[i, l] != y_train[l]:
                D[i, l] = D[i, l]/(2 * (error+0.001))
            else:
                D[i, l] = D[i, l] / (2 * (1-error))
        
    # vote -- 根据权重对标签投票，最多的则为预测结果
    result = np.zeros(num_x2)
    for i in range(num_x2):
        vote_0 = 0
        vote_1 = 0
        vote_2 = 0
        for j in range(m):
            if class_result_test[j, i] == 0:
                vote_0 += alpha[j]
            elif class_result_test[j, i] == 1:
                vote_1 += alpha[j]
            else:
                vote_2 += alpha[j]
        vote = [vote_0, vote_1, vote_2]
        result[i] = vote.index(max(vote))
        
    result = np.int64(result)
    accuracy = accuracy_score(result, y_test)
    
    # 返回分类结果
    return result, accuracy

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
    # boosting(knn)
    result2, accuracy2 = boosting_knn(x_train, y_train, x_test, y_test, k=11, m=10, n=5)
    print('knn(k=11)_accuracy:%.3f'%accuracy1)
    print('boosting_10个knn(k=11)_accuracy:%.3f'%accuracy2)
    
     # svm
    kernel=['linear', 'poly', 'rbf', 'sigmoid']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_rate=0.368)
    svm = SVM.SVC(C=1, kernel=kernel[0], decision_function_shape='ovr')
    svm.fit(x_train, y_train)
    result3 = svm.predict(x_test)
    accuracy3 = accuracy_score(result3, y_test)
    # boosting(svm)
    result4, accuracy4 = boosting_svm(x, y, m=10, n=5, kernel=kernel[0])
    print('svm_accuracy:%.3f'%accuracy3)
    print('boosting_10个svm_accuracy:%.3f'%accuracy4)






