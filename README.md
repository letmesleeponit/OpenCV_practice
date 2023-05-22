# OpenCV Examples.
## Guide
---
This repository contains several programs that I have developed while practicing OpenCV applications. Currently, it includes the following projects:
1. Credit Card Digit Detection
2. OCR(Work in progress)

## Results:
---
### 1. Credit Card Number Detection
1. Capture number templates (1-9) to be used for matching credit card numbers.
2. Perform edge detection to identify four consecutive and bigger bounding boxes on the credit card, and extract the digits from each box.
3. Utilize the cv2.matchTemplate function to compare the templates with the extracted digits, selecting the one with the highest accuracy as the output.  
<img decoding="async" src="Credit Card Number Detection.png" width="30%">

### 2. Optical Character Recognition
1. Capture the corners of the document.
2. Apply  Perspective transform to map the document to a front-facing perspective.
3. Utilize pytesseract for OCR  
<img decoding="async" src="OCR_Perspective transformation.png" width="30%">

