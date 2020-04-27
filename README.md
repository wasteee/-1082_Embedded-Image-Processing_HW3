# -1082_Embedded-Image-Processing_HW3

作業3:找一張馬路上的圖片，用紋路(LBP)來區分出馬路與街景的分割。

# 設計 </p>

整體流程圖如下</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/others/model.png)

1.先將圖片切割出馬路與不是馬路的部分。</p>
2.再將每個區塊切割成 kernel 10x10，並計算每個 kernel 中的 LBP, Gray, Hue 和 Haralick texture ，然後統計中前三者的平均數、中位數、眾數和標準差，將其作為特徵，因此特徵共有 3x4 + 13(Haralick texture) ，共25個。</p>
3.使用 KNN 與 Decision Tree 訓練，分辨出每個 kernel 是否為柏油路，k 設定為 3 。</p>
4.將訓練完成後的模組用來判別，並將結果去雜訊。</p>

# 結果 </p>
- 總共切割出 10 張非馬路與 22 張馬路的照片，共 11 萬筆特徵。</p>
- 將特徵使用PCA分析後，在第 12 筆資料時就達到 98%~99% ，因此可特徵數可降至 12 左右即可。</p>
- 訓練結果KNN 準確率可達 0.97 ， Decision Tree 準確率可達 0.98。</p>
- 由於辨識後的結果會有許多雜訊，因此去雜訊時會判別若中間點的連通皆為 255 或 0 時，中間點就調整為 255 或 0 ，經過測試四連通的效果比八連通更好。</p>

- 原圖1</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/fullroad/road9.jpg)
- 結果圖1，去雜訊前 ，其中白色部分為馬路</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/outputs/Final_f_dt_v2_p9_b.jpg)
- 結果圖1，去雜訊後</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/outputs/Final_f_dt_v2_p9_a.jpg)
- 原圖2</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/fullroad/road13.jpg)
- 結果圖2，去雜訊前 ，其中白色部分為馬路</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/outputs/Final_f_dt_v2_p11_b.jpg)
- 結果圖2，去雜訊後</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/outputs/Final_f_dt_v2_p11_a.jpg)
- 原圖3</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/fullroad/road4.jpg)
- 結果圖3，去雜訊前 ，其中白色部分為馬路</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/outputs/Final_f_dt_v2_p12_b.jpg)
- 結果圖3，去雜訊後</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/outputs/Final_f_dt_v2_p12_a.jpg)
- 原圖4</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/fullroad/road12.jpg)
- 結果圖4，去雜訊前 ，其中白色部分為馬路</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/outputs/Final_f_dt_v2_p13_b.jpg)
- 結果圖4，去雜訊後</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/outputs/Final_f_dt_v2_p13_a.jpg)
- 原圖5 (無出現於訓練資料中)</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/fullroad/road21.jpg)
- 結果圖5，去雜訊前 ，其中白色部分為馬路</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/outputs/Final_f_dt_v2_p14_b.jpg)
- 結果圖5，去雜訊後</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/outputs/Final_f_dt_v2_p14_a.jpg)

# 總結 </p>
整體而言，柏油路的部分都可以正確的辨識出來，但是遇到樹林的時候由於樹林的紋路與柏油路的紋路相近，導致會有些許誤判的出現，另外由於遠處的柏油對焦較模糊，所以辨識時也會辨識失敗，另外此作法還有一個缺點就是整體的執行速度非常緩慢，一張 1920x1280 的圖需要 2~3 分鐘才可辨識完成，整體實用性較低。
