'''
conda install opencv
'''

import imutils
import cv2
import numpy as np
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

image = cv2.imread("4.png")
(h, w, d) = image.shape #읽어들인 이미지 파일의 해상도
print("width={}, height={}, depth={}".format(w, h, d))
cv2.imshow("Image", image)

img = imutils.resize(image, width=500 ) # 이미지 가로폭 해상도 500설정
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #컬러이미지 흑백 변환
gray = cv2.bilateralFilter(gray, 11, 17, 17) #노이즈 축소를 위한 blurring
edged = cv2.Canny(gray, 30, 200) # 30~200 밝기를 가지는 Canny Edge 필터링
cv2.imshow("Canny",edged)
cv2.imwrite('GrayEdge_Car.png',edged)

#번호판 인식을 위해 Canny Edge 처리된 이미지에서 Contour들을 찾아 draw해준다
cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img1=img.copy()
cv2.drawContours(img1,cnts,-1,(0,255,0),3)
cv2.imshow("img1",img1)

cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #Contours중 면적 기준 큰 순서대로 30개 draw
screenCnt = None #번호판 윤곽 저장
img2 = img.copy()
cv2.drawContours(img2,cnts,-1,(0,255,0),3)
cv2.imshow("img2",img2)

idx=7
# loop over contours

for c in cnts:
  # contour 근사치내기
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4: #chooses contours with 4 corners
                screenCnt = approx
                x,y,w,h = cv2.boundingRect(c) #finds co-ordinates of the plate
                new_img=img[y:y+h,x:x+w]
                cv2.imwrite('./licenseplate.png',new_img) #stores the new image
                break

cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("Final image with plate detected",img) #번호판 Contour 출력
cv2.imwrite("LicensePlate_Car.png",img)