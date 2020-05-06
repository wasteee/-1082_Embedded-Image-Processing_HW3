# 1082_Embedded-Image-Processing_HW3

作業3:找一張馬路上的圖片，用紋路(LBP)來區分出馬路與街景的分割。

# 設計 </p>
- 方法一 Machine learning</p>
整體流程圖如下</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/others/model.png)

1.先將圖片切割出馬路與不是馬路的部分。</p>
2.再將每個區塊切割成 kernel 10x10，並計算每個 kernel 中的 LBP, Gray, Hue 和 Haralick texture ，然後統計中前三者的平均數、中位數、眾數和標準差，將其作為特徵，因此特徵共有 3x4 + 13(Haralick texture) ，共25個。</p>
3.使用 KNN 與 Decision Tree 訓練，分辨出每個 kernel 是否為柏油路，k 設定為 3 。</p>
4.將訓練完成後的模組用來判別，並將結果去雜訊。</p>


- 方法二 BFS區域生成法</p>
1.由左下選擇一點(kernel 40x40)出發</p>
2.將目前所在點的輸出設為 255 ，並記錄已造訪</p>
3.若相鄰點的 LBP 直方圖與 Hue 直方圖與目前點相識度在閥值內，則將下一點的輸出調為 255 並造訪；若不在閥值內，則記錄已造訪</p>
4.重複2-3步驟直到所相鄰便都造訪完成</p>

# 方法一結果 </p>
- 總共切割出 10 張非馬路與 22 張馬路的照片，共 11 萬筆特徵。</p>
- 將特徵使用PCA分析後，在第 12 筆資料時就達到 98%~99% ，因此可特徵數可降至 12 左右即可。</p>
- 訓練結果KNN 準確率可達 0.97 ， Decision Tree 準確率可達 0.98。</p>
- 由於辨識後的結果會有許多雜訊，因此去雜訊時會判別若中間點的連通皆為 255 或 0 時，中間點就調整為 255 或 0 ，經過測試四連通的效果比八連通更好。</p>

# 方法二結果
由於圖片中的左下與右下皆較容易是柏油路，所以從這些點開始往外做BFS會是一個很好的選擇，而以下結果則皆由左下開始</p>

- 原圖1</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/fullroad/road9.jpg)
- 方法一結果圖1，/合成圖</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/outputs/0com.jpg)
- 方法二結果圖1，/合成圖</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/gen/5gen.jpg)
- 原圖2</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/fullroad/road13.jpg)
- 方法一結果圖2，/合成圖</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/outputs/1com.jpg)
- 方法二結果圖2，/合成圖</p>
![image](https://github.com/wasteee/-1082_Embedded-Image-Processing_HW3/blob/master/image/gen/1gen.jpg)




# 總結 </p>
整體而言，柏油路的部分都可以正確的辨識出來，但是遇到樹林的時候由於樹林的紋路與柏油路的紋路相近，導致會有些許誤判的出現，另外由於遠處的柏油對焦較模糊，所以辨識時也會辨識失敗，這些作法還有一個缺點就是整體的執行速度非常緩慢，一張 1920x1280 的圖需要 2~3 分鐘才可辨識完成，整體實用性較低。
