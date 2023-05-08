import os
import cv2
import numpy as np

### 讀取信用卡圖片
pth = os.path.dirname(os.path.abspath(__file__))
img_rgb = cv2.imread(os.path.join(pth, 'data1.jpg'))
img_rgb = cv2.resize(img_rgb, (1024, 512))
img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
img_copy = img.copy()

### 透過邊緣檢測找到每個數字對應的像素，並將其當成模板，用來檢驗信用卡上面的數字
digit = {} # 儲存模板的x,y,w,h資訊
digit_templates = {} # 儲存模板的像素資訊
for model_name in ['01234.png', '56789.png']:
    img_digit = cv2.imread(os.path.join(pth, model_name), 0)
    img_digit_copy = img_digit.copy() # 用來畫圖
    canny = cv2.Canny(img_digit, 150, 200)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    res = cv2.drawContours(img_digit_copy, contours, -1, 255, 2)
    digit_temp = {} # 暫時儲存每張照片模板的像素資訊
    ### 紀錄這些數字的位置，並將其像素存在digit中當作模板
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_digit_copy, (x, y), (x + w, y + h), 255, 2)
        digit_temp[str(x)] = [x, y, w, h]
    ### 按照在圖上的位置進行排序，並更新key value為該模板對應數字
    for key, new_key in zip(sorted(list(digit_temp.keys())), model_name.split('.')[0]):
        digit[new_key] = digit_temp[key]
        digit_templates[new_key] = img_digit[digit[new_key][1]:digit[new_key][1]+digit[new_key][3], digit[new_key][0]:digit[new_key][0]+digit[new_key][2]]
        digit_templates[new_key] = cv2.resize(digit_templates[new_key], (150, 150))
        cv2.putText(img_digit_copy, new_key, (int(digit[new_key][0] + digit[new_key][2] / 2), digit[new_key][1]), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2) # 在圖上寫下對應數字

### 透過開運算, 高斯濾波器等處理讓雜訊盡量減少，並讓數字盡量擠成一塊
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 4))
open = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
ret, res = cv2.threshold(open, 140, 255, cv2.THRESH_BINARY)
blur = cv2.GaussianBlur(res, (5, 5), 5)
open = cv2.morphologyEx(blur, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
dilate = cv2.dilate(open, np.ones((5, 5), np.uint8), iterations=5)
open = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=5)
ret, res = cv2.threshold(open, 127, 255, cv2.THRESH_BINARY)
close = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=5)

### 抓取信用卡中數字的邊框
canny = cv2.Canny(close, 150, 200)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = [(x, y, w, h) for contour in contours for (x, y, w, h) in [cv2.boundingRect(contour)] if w > 100] # 把長度太小的框給去掉，因為一定不可能是框住數字的

### 剩下的這些框中，找y相近，且面積相近者
length = 5 # 用來抓那4個框的時候，往外拓展一些空間
for ind in range(len(contours)):
    contours_one_d = [contour[1] for contour in contours]
    result = np.logical_and([contours[ind][1] + length >= contour_one_d for contour_one_d in contours_one_d], [contours[ind][1] - length <= contour_one_d for contour_one_d in contours_one_d])
    # 符合y座標接近的框有4個
    if np.sum(result) == 4: 
        contours = np.array(contours)[np.where(result)[0]]
        break
    # 符合y座標接近的框有大於4個，則抓面積相近的四個
    elif np.sum(result) >= 4: 
        contours = np.array(contours)[np.where(result)[0]]
        area = [contour[2] * contour[3] for contour in contours]
        ### 透過按照大小減差值，接著將這些連續的差值進行相加取最小值的方式找面積最相近的四個方框
        area_sorted = sorted(area)
        diffs_area_sorted = [area_sorted[ind+1] - area_sorted[ind] for ind in range(len(area)-1)]
        diffs_ind = np.argmin([sum(diffs_area_sorted[ind:ind+3]) for ind in range(len(diffs_area_sorted))])
        indexes = []
        for ele_area_sorted in area_sorted[diffs_ind:diffs_ind+4]:
            indexes.append(area.index(ele_area_sorted))
        contours = contours[indexes]
    elif ind == len(contours) -1 :
        raise SystemExit("無法完整偵測到含有數字的四個方格，需要調整影像前處理參數")

### 另存信用卡上數字的像素到list裡
img_card_digits_gray = [] # gray資訊，剪取個別數字的位置最後用來和模板比較
img_card_digits = [None] * 4 # 複製gray資訊，用來抓取個別數字的位置
for contour in contours: # 紀錄信用卡中四個含有數字大方框的位置
    img_card_digits_gray.append(img[contour[1]-length:contour[1]+contour[3]+length, contour[0]-length:contour[0]+contour[2]+length])

digit_predict = {} # 儲存最終結果
### 對這些數字進行前處理
for ind in range(len(img_card_digits_gray)):
    img_card_digits[ind] = cv2.morphologyEx(img_card_digits_gray[ind], cv2.MORPH_TOPHAT, np.ones((5, 5), np.uint8), iterations=2)
    ret, img_card_digits[ind] = cv2.threshold(img_card_digits[ind], 64, 255, cv2.THRESH_BINARY)
    img_card_digits[ind] = cv2.morphologyEx(img_card_digits[ind], cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    img_card_digits[ind] = cv2.Canny(img_card_digits[ind], 125, 200)
    digit_countours, digit_hierarchy = cv2.findContours(img_card_digits[ind], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ### 分別抓取每個含有數字的大方框內部數字
    number_indices = np.argsort([cv2.boundingRect(digit_countour)[3] for digit_countour in digit_countours])[-4:] # 抓方格內面積最大的4個物件(因為可能會抓到像8內部的圓等雜訊)
    for number_index in number_indices:
        res = cv2.drawContours(img_card_digits[ind], digit_countours[number_index], -1, 255, 2)
        x, y, w, h = cv2.boundingRect(digit_countours[number_index])
        digit_img = img_card_digits_gray[ind][y:y+h, x:x+w]
        digit_img = cv2.resize(digit_img, (150, 150))
        ret, digit_img = cv2.threshold(digit_img, 127, 255, cv2.THRESH_BINARY_INV)
        result = {}
        ### 測試每個模板，抓取matchTemplate分數最高的圖形
        for templates_ind, digit_template in digit_templates.items():
            res = cv2.matchTemplate(digit_img, digit_template, cv2.TM_CCOEFF_NORMED)
            result[templates_ind] = res[0][0]
        digit_predict[x + contours[ind][0]] = int(max(result, key=result.get)) 

### 按照在圖上的位置進行排序，並更新key value為該模板對應數字
prdict = ''
for key in sorted(list(digit_predict.keys())):
    prdict += str(digit_predict[key])

### 結果呈現
cv2.imshow("img", img_rgb)
print('預測結果為', prdict)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()