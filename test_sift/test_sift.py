# coding: utf-8
import cv2, cv2 as cv
import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import numpy as np

img_path = "./data_yegan/set_01/_1018843.JPG"
log.info( f"img_path = {img_path}" )

img = cv.imread( img_path )
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp, descriptors = sift.detectAndCompute(gray, None)

#img=cv.drawKeypoints(gray, kp, img)
img=cv.drawKeypoints(gray, kp, img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

scale_percent = 10  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("Image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()