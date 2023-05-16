import os
import numpy as np
import cv2
import func

dir_pth = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(dir_pth, 'img_1.jpg'))
img = cv2.resize(img, (1024, 768))
img_copy = img.copy()
### 圖像前處理，方便之後抓取輪廓
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0) # 過濾雜訊
canny = cv2.Canny(img_gray, 150, 200)
img_new = cv2.dilate(canny, (np.ones((5, 5), np.uint8))) # 膨脹邊緣，不然發票邊緣太細，findContours會無法抓到

### 找到要進行OCR的文件輪廓
# 由面積最大開始往面積小的找，同時這個面積近似後的點要是4個點(矩形)
cnts, _ = cv2.findContours(img_new, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5] # 抓取面積最大的前 n 個輪廓
for cnt in cnts:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    # 由最大面積開始往下篩，四個角點(矩形)就跳出
    if len(approx) == 4:
        file_cnt = approx
        break

### 進行透視
# 取得原本文件座標(四個角點)
rect = func.get_four_points(approx) # 取得原先文件位置
width = max(int(np.sqrt(sum((rect[1] - rect[0]) ** 2))), int(np.sqrt(sum((rect[3] - rect[2]) ** 2)))) # 取原先文件寬度最大值作為轉換尺度
height = max(int(np.sqrt(sum((rect[2] - rect[0]) ** 2))), int(np.sqrt(sum((rect[3] - rect[1]) ** 2)))) # 取原先文件長度最大值作為轉換尺度
dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32') # 要轉換過去的座標
M = cv2.getPerspectiveTransform(rect, dst) # 得到透視矩陣
warped = cv2.warpPerspective(img_copy, M, (width, height)) # 輸入原本文件位置及透視矩陣得到新的矩陣
warped = cv2.resize(warped, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('warped', warped)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()

### 偵測文字
# to be continued...