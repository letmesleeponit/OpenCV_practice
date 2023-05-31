# OpenCV Examples.
## Guide
---
This repository contains several programs that I have developed while practicing OpenCV applications. Currently, it includes the following projects:
1. Credit Card Digit Detection
2. Optical Character Recognition
3. Image Stiching

## Results:
---
### 1. Credit Card Number Detection
1. Capture number templates (1-9) to be used for matching credit card numbers.
2. Perform edge detection to identify four consecutive and bigger bounding boxes on the credit card, and extract the digits from each box.
3. Utilize the cv2.matchTemplate function to compare the templates with the extracted digits, selecting the one with the highest accuracy as the output.  
<img decoding="async" src="信用卡數字檢測/Credit Card Number Detection.png" width="30%">

### 2. Optical Character Recognition(OCR)
1. Capture the corners of the document.
2. Apply  Perspective transform to map the document to a front-facing perspective.
3. Utilize pytesseract for OCR  
<img decoding="async" src="OCR/OCR_Perspective transformation.png" width="30%">

### 3. Image Stiching
1. Use the SIFT algorithm to detect the mutual features in two images.
2. Utilize the brute-force matcher to compare the similarity between features in the two images.
3. Obtain the transformation matrix by extracting corresponding features from the two images.
4. Transform the images and merge them together.  
<img decoding="async" src="圖像拼接/brute_force_match.png" width="30%">
<img decoding="async" src="圖像拼接/result.png" width="30.22%">
