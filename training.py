# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:05:59 2020

@author: waster
"""
import csv
import pickle
import numpy as np
from os import listdir
from os.path import isfile, join
np.set_printoptions(threshold=np.inf)
list_x = []
list_y = []
with open('G:\\imagetest\\file_6f.csv', newline='') as csvfile:

    # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)

    for row in rows:
        list_x.append(row[:-1])
        list_y.append(row[6:7])
list_x = list_x[1:]
list_y = list_y[1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(list_x,list_y,test_size = 0.1,random_state = 41)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)


#
#from sklearn.svm import SVC
#svm = SVC(gamma='auto')
#svm.fit(x_train, y_train)
yknn_bef_scaler = knn.predict(x_test)
acc_knn_bef_scaler = accuracy_score(yknn_bef_scaler,y_test)

print('data len : ' , len(list_y))
print(acc_knn_bef_scaler)


filename = "G:\\imagetrain\\hw3_knn_6f_v0.sav"
pickle.dump(knn,open(filename,"wb"))