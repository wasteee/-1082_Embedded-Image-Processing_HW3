# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 21:26:48 2020

@author: waster
"""

import pickle
import csv
import mahotas
from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import stats
def statistics(img,name,img_gray,name_gray,img_h):
    output = []
    output.append(np.mean(img))
    output.append(np.std(img,ddof=1))
    output.append(np.median(img))
    output.append(stats.mode(img).mode[0][0])
    output.append(np.mean(img_gray))
    output.append(np.std(img_gray,ddof=1))
    output.append(np.median(img_gray))
    output.append(stats.mode(img_gray).mode[0][0])
    output.append(np.mean(img_h))
    output.append(np.std(img_h,ddof=1))
    output.append(np.median(img_h))
    output.append(stats.mode(img_h).mode[0][0])
    img_gray = img_gray.astype(int)
    haralick = mahotas.features.haralick(img_gray).mean(0)
    for i in range(0,13):
        output.append(haralick[i])

    
    return output
def imgfilter(img,height, width):
    img123 = img
    height, width = int(height/10) - height%10, int(width/10) - width%10
    for i in range(1,height-1):
        for j in range(1,width-1):
#            if(img[i-1,j] >200 and img[i-1,j+1] >200 and img[i-1,j-1] >200 and img[i,j-1] >200 \
#               and img[i,j+1] >200 and img[i+1,j-1] >200 and img[i+1,j] >200 and img[i+1,j+1] >200):
#                img123[i,j] = 255
#            elif(img[i-1,j] <200 and img[i-1,j+1] <200 and img[i-1,j-1] <200 and img[i,j-1] <200 \
#               and img[i,j+1] <200 and img[i+1,j-1] <200 and img[i+1,j] <200 and img[i+1,j+1] <200):
#                 img123[i,j] = 0
            if(img[i-1,j] >200 and img[i,j-1] >200 \
               and img[i,j+1] >200 and img[i+1,j] >200 ):
                img123[i,j] = 255
            elif(img[i-1,j] <200  and img[i,j-1] <200 \
               and img[i,j+1] <200  and img[i+1,j] <200 ):
                 img123[i,j] = 0
    
    return img123
np.set_printoptions(threshold=np.inf)
filename = "G:\\imagetrain\\hw3_dt_f_v1.sav"
model = pickle.load(open(filename,'rb'))

radius = 1  # LBP算法中范围半径的取值
n_points = 8 # 领域像素点数
# 读取图像
image1 =cv2.imread('G:\\imagetest\\fullroad\\road12.jpg',1)
height, width, channels = image1.shape
#cv2.imshow("image1",image1)
hsv_img = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
#plt.imshow(h)
#plt.show()
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray1 = gray



lbp = local_binary_pattern(image1, n_points, radius,method = 'ror')
lbp = lbp / 16


k = np.array(np.zeros((10,10)))
k_gray = np.array(np.zeros((10,10)))
k_h = np.array(np.zeros((10,10)))
image_new = np.array(np.zeros((int(height/10) - height%10,int(width/10) - width%10)))

for col in range(0,int(height/10) - height%10):
    for row in range(0,int(width/10) - width%10):
        k = np.round(lbp[col*10:col*10 + 10,row*10:row*10 + 10])               
        k_gray = gray1[col*10:col*10 + 10,row*10:row*10 + 10]
        k_h = h[col*10:col*10 + 10,row*10:row*10 + 10]
        
        inp_temp = statistics(k,'k',k_gray,'k_gray',k_h)
        inp_temp = np.array(inp_temp).reshape(1, -1)
        seg_type = model.predict(inp_temp)
        if(seg_type == '1'):
            image_new[col,row] = 255
        else:
            image_new[col,row] = 0


plt.imshow(image_new)
cv2.imwrite("G:\\imagetest\\Final_f_dt_v1_p7_b.jpg",image_new)
plt.show()
image_new = imgfilter(image_new,height, width)
#Final = cv2.addWeighted(gray1,0.5,image1,0.5,0)
plt.imshow(image_new)
plt.show()
cv2.imwrite("G:\\imagetest\\Final_f_dt_v1_p7_a.jpg",image_new)


cv2.waitKey(0)
cv2.destroyAllWindows()