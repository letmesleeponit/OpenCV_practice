import os
import cv2
import numpy as np

### 讀取影像位置，這邊統一設定圖像長為600
dir_path = os.path.dirname(os.path.abspath(__file__))
img1 = cv2.imread(os.path.join(dir_path, 'img_1.jpg'))
ratio = img1.shape[0] / 600
img1 = cv2.resize(img1, (int(img1.shape[1]/ratio), 600))
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(os.path.join(dir_path, 'img_2.jpg'))
ratio = img2.shape[0] / 600
img2 = cv2.resize(img2, (int(img2.shape[1]/ratio), 600))
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

### 使用Sift算法找到圖像特徵點
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)
bf = cv2.BFMatcher()

### 一對多匹配
matches = bf.knnMatch(des1, des2, k=2)
matches = sorted(matches, key=lambda x: x[0].distance)[:4]

# ### 透過比較knnMatch對應的多個特徵點中，對應到的多個特徵點距離差異是否夠大，來確認這個特徵是否足夠明顯
match_point = [] # 紀錄符合的座標
for match in matches:
    if len(match) ==2 and match[0].distance < match[1].distance * 0.75: # 官方文檔用0.75
        match_point.append((match[0].trainIdx, match[0].queryIdx))

# ### 至少要有四個座標才能求變換矩陣
if len(matches) >= 4:
    ptsA = np.float32([kp1[i].pt for (_, i) in match_point])
    ptsB = np.float32([kp2[i].pt for (i, _) in match_point])
    ptsB = np.float32([[ptB[0] + img1.shape[1], ptB[1]] for ptB in ptsB]) # 把寬度設定為原本的圖片加上要合併圖片的寬度
    M, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
    result = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))

result[:, (result.shape[1] - img2.shape[1]):result.shape[1]] = img2

cv2.imshow('result', result)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()