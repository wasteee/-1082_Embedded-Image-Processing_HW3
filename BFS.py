import pickle
import csv
import mahotas
from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import stats

def comparelbp(height1,width1,height2,width2,img,h,rate1,rate2):
    radius = 1 
    n_points = 8
    image1 = img[height1*40:height1*40 + 40,width1*40:width1*40 + 40]
    h1 = h[height1*40:height1*40 + 40,width1*40:width1*40 + 40]
    lbp1 = local_binary_pattern(image1, n_points, radius,method = 'ror')
    image2 = img[height2*40:height2*40 + 40,width2*40 :width2*40 + 40]
    h2 = h[height1*40:height1*40 + 40,width1*40:width1*40 + 40]
    lbp2 = local_binary_pattern(image2, n_points, radius,method = 'ror')
    H1 = cv2.calcHist([lbp1.astype('uint8')], [0], None, [256],[1,254])
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1) 
    H2 = cv2.calcHist([lbp2.astype('uint8')], [0], None, [256],[1,254])
    H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)
    similarity1 = cv2.compareHist(H1, H2, cv2.HISTCMP_CORREL)
    H1 = cv2.calcHist([h1.astype('uint8')], [0], None, [256],[1,254])
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1) 
    H2 = cv2.calcHist([h2.astype('uint8')], [0], None, [256],[1,254])
    H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)
    similarity2 = cv2.compareHist(H1, H2, cv2.HISTCMP_CORREL)
    print(similarity1,similarity2)
    if(similarity1 > rate1 and similarity2 > rate2):
        return True
    return False

class note():
    def __init__(self, n, m):
        self.x = [0] * n * m  # 縱座標
        self.y = [0] * n * m  # 橫座標
        self.f = [0] * n * m  # 父親在佇列中的編號
        self.s = [0] * n * m  # 步數

str1 = ['','','','','','','','','','','','','','','']
str1[0] = 'G:\\imagetest\\fullroad\\lbp.jpg'
str1[1] = 'G:\\imagetest\\fullroad\\road13.jpg'
str1[2] = 'G:\\imagetest\\fullroad\\road4.jpg'
str1[3] = 'G:\\imagetest\\fullroad\\road12.jpg'
str1[4] = 'G:\\imagetest\\fullroad\\road21.jpg'
str1[5] = 'G:\\imagetest\\fullroad\\road9.jpg'
str1[6] = 'G:\\imagetest\\fullroad\\road1_.jpeg'
str1[7] = 'G:\\imagetest\\fullroad\\road7.jpeg'
str1[8] = 'G:\\imagetest\\fullroad\\road8.jpeg'
str1[9] = 'G:\\imagetest\\fullroad\\road10.jpg'
str1[10] = 'G:\\imagetest\\fullroad\\road15.jpg'
str1[11] = 'G:\\imagetest\\fullroad\\road16.jpg'
str1[12] = 'G:\\imagetest\\fullroad\\road17.jpg'
str1[13] = 'G:\\imagetest\\fullroad\\road18.jpg'
str1[14] = 'G:\\imagetest\\fullroad\\road19.jpg'
for qwe in range(0,15):
    image1 =cv2.imread(str1[qwe],0)
    image2 =cv2.imread(str1[qwe],0)
    height, width = image1.shape
    hsv_img = cv2.imread(str1[qwe],1) 
    hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    
    
    # 此為迷宮，0 = 空格，1 = 障礙
    maze = np.array(np.zeros((int(height/40),int(width/40))))
    outputs = np.array(np.zeros((int(height/40),int(width/40))))
    print(np.shape(outputs))
    # 設定迷宮大小
    n = len(maze)
    m = len(maze[0])
    # 設一個陣列，用來標記走過的座標
    book = np.array(np.zeros((int(height/40),int(width/40))))
    print(np.shape(book))
    #for i in range(n):
    #    book.append([0] * m)
    #print(np.shape(book))
    # 設定起訖點
    startx = int((height/40)*(9/10))
    starty = int((width/40)*(7/10))
    print(startx,starty)
    outputs[startx][starty] = 255
    endx = -1
    endy = -1
    
    # 定義一個表示走的方向的陣列
    next = [[0, 1],   # 向右走
            [1, 0],   # 向下走
            [0, -1],  # 向左走
            [-1, 0]]  # 向上走
    
    # 佇列初始化
    head = 0
    tail = 0
    
    # 往佇列插入迷宮入口座標
    que = note(n, m)
    que.x[tail] = startx
    que.y[tail] = starty
    que.f[tail] = 0
    que.s[tail] = 0
    tail += 1
    #book[startx][startx] = 1
    
    flag = 0 # 用來標記是否到達目標，0 = 未到，1 = 到達
    
    # 當佇列不為空的迴圈
    while(head < tail):
        # 列舉4個方向
        for i in range(4):
            # 計算下一個座標
            tx = que.x[head] + next[i][0]
            ty = que.y[head] + next[i][1]
            # 判斷是否越界
            if tx < 0 or tx > n-1 or ty < 0 or ty > m-1 :
                continue
            # 判斷是否是障礙物或者已經走過
            plt.imshow(outputs)
            plt.show()
            if comparelbp(tx,ty,que.x[head],que.y[head],image1,h,0.985,0.99) and book[tx][ty] == 0 :
                book[tx][ty] = 1  # 標記為已走過
                outputs[tx][ty] = 255
                # 插入新的點到佇列中
                que.x[tail] = tx
                que.y[tail] = ty
                que.f[tail] = head
                que.s[tail] = que.s[head] + 1   # 步數是父親步數+1
                tail += 1
            # 若到訖點，停止擴展退出迴圈
            else:
                book[tx][ty] = 1
            if tx == endx and ty == endy :
                flag = 1
                break
        if flag == 1 :
            break
        head += 1 # 當一個擴展結束後，要head++才能對後面的點再進行擴展
    
    # 列印佇列中末尾最後一個點(訖點)的步數，但tail是指向佇列尾的下一個位置，所以要-1
    #print(que.s[tail-1])
    #print(que.x)
    #print(que.y)
    #print(flag)
    plt.imshow(image1)
    plt.show()
    for col in range(0,int(height/40)):
        for row in range(0,int(width/40)):
            image1[col*40:col*40 + 40,row*40:row*40 + 40] = outputs[col,row]
    Final = cv2.addWeighted(image1,0.5,image2,0.5,0)
    t = "G:\\imagetest\\gen\\" + str(qwe) + "newgen.jpg"
    plt.imshow(Final)
    plt.show()
    cv2.imwrite(t,Final)

