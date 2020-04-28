# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 00:51:33 2020

@author: waster
"""
import csv
import mahotas
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io,data_dir,filters, feature
from skimage.color import label2rgb
from sklearn.metrics import mean_squared_error
import skimage
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy import stats

def statistics(img,name,img_h,name_h,img_g):
    arr_mean = np.mean(img)
    arr_std = np.std(img,ddof=1)
    median = np.median(img)
    mode = stats.mode(img).mode[0][0]
    arr_mean_h = np.mean(img_h)
    arr_std_h = np.std(img_h,ddof=1)
    median_h = np.median(img_h)
    mode_h = stats.mode(img_h).mode[0][0]
    arr_mean_gray = np.mean(img_g)
    arr_std_gray = np.std(img_g,ddof=1)
    median_gray = np.median(img_g)
    mode_gray = stats.mode(img_g).mode[0][0]
    img_g = img_g.astype(int)
    haralick = mahotas.features.haralick(img_g).mean(0)
    with open('G:\\imagetest\\file_full4.csv', 'a', newline='') as csvfile:
      # 建立 CSV 檔寫入器
          writer = csv.writer(csvfile)
          writer.writerow([arr_mean, arr_std, median, mode,arr_mean_gray,arr_std_gray, \
                       median_gray,mode_gray,arr_mean_h,arr_std_h,median_h,mode_h, \
                       haralick[0],haralick[1], \
                       haralick[2],haralick[3],haralick[4],haralick[5],haralick[6], \
                       haralick[7],haralick[8],haralick[9],haralick[10],haralick[11] \
                       ,haralick[12],1])


#def hisMSE(img1,img2):
#    img1_his = np.array(np.zeros((256)))
#    img2_his = np.array(np.zeros((256)))
#    unique, counts = np.unique(img1, return_counts=True)
#    img1_dict = dict(zip(unique, counts))
#    unique, counts = np.unique(img2, return_counts=True)
#    img2_dict = dict(zip(unique, counts))
#    for i in range(0,256):
#        if(i in img1_dict):
#            img1_his[i] = (img1_dict[i])
#        if(i in img2_dict):
#            img2_his[i] = (img2_dict[i])
#    mse = mean_squared_error(img1_his, img2_his)
#    return mse
            

np.set_printoptions(threshold=np.inf)
# settings for LBP
radius = 1  
n_points = 8 

image1 =cv2.imread('G:\\imagetest\\road\\road9_t.jpg',1)
height, width, channels = image1.shape
hsv_img = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
#cv2.imshow("image1",image1)
plt.imshow(h)
plt.show()
plt.hist(image1.ravel(), 255, [0, 256])
plt.show()

gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray1 = gray
ret,gray = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)#二值化
gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
#cv2.imshow("gray1",gray1)
#cv2.imwrite("G:\\imagetest\\image123.jpg",gray1)

# 畫出直方圖
plt.hist(gray.ravel(), 255, [0, 256])
plt.show()


lbp = local_binary_pattern(image1, n_points, radius,method = 'ror')

lbp = lbp / 16

k = np.array(np.zeros((10,10)))
k_gray = np.array(np.zeros((10,10)))
k_h = np.array(np.zeros((10,10)))
#k1 = np.array(np.zeros((10,10)))
#k1_gray = np.array(np.zeros((10,10)))
for col in range(0,int(height/10) - height%10):
    for row in range(0,int(width/10) - width%10):
        k = np.round(lbp[col*10:col*10 + 10,row*10:row*10 + 10])               
        k_gray = gray1[col*10:col*10 + 10,row*10:row*10 + 10]
        k_h = h[col*10:col*10 + 10,row*10:row*10 + 10]

        statistics(k,'k',k_h,'k_gray',k_gray)






plt.hist(lbp.ravel(), 255, [0, 255])
plt.show()
plt.hist(k.ravel(), 255, [0, 255])
plt.show()
#plt.hist(k1.ravel(), 255, [0, 15])
#plt.show()
#print('MSE',hisMSE(k1,k))
#edges = filters.sobel(image1)
#cv2.imshow("edges",edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
