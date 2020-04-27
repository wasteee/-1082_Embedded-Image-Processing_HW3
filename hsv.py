# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 04:26:54 2020

@author: waster
"""
import cv2
import matplotlib.pyplot as plt
hsv_img = cv2.imread('G:\\imagetest\\fullroad\\road11.jpg',1) 
hsv_img = hsv_img
hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)
h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
#image0 = cv2.cvtColor(image0, cv2.COLOR_HSV2GRAY)
plt.imshow(h)
plt.show()
plt.imshow(s)
plt.show()
plt.imshow(v)
plt.show()
plt.hist(h.ravel(), 256, [0, 255])
plt.show()
#v = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
#plt.imshow(v)
#plt.show()