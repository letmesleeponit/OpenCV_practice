import os
import cv2 

dir_pth = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(dir_pth, 'img_1.jpg'))
img = cv2.resize(img, (1024, 768))
img_copy = img.copy()
### 圖像前處理，方便之後抓取輪廓
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0) # 過濾雜訊
canny = cv2.Canny(img_gray, 150, 200)
### 找到要進行OCR的文件輪廓
# 由面積最大開始往面積小的找，同時這個面積近似後的點要是4個點(矩形)
cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = sorted(cnts, key=cv2.contourArea)[-5:] # 抓取面積最大的前 n 個輪廓
for cnt in cnts:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    cv2.drawContours(img_copy, approx, -1, (0, 0, 255), 2)
    cv2.imshow('img_copy', img_copy)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()


    if len(approx) == 4:
        file_cnt = approx
        break

cv2.drawContours(img_copy, file_cnt, -1, (0, 0, 255), 2)
cv2.imshow('img_copy', img_copy)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()