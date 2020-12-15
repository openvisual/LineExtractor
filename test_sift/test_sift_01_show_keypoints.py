# coding: utf-8

import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

print( "OpenCV version : %s" % cv.__version__ )

print( "Pwd 1: %s" % os.getcwd())
# change working dir to current file
dirname = os.path.dirname(__file__)
dirname and os.chdir( dirname )
dirname and print( "Pwd 2: %s" % os.getcwd())

img_path = "./data_yegan/set_00/01_left.jpg"
#img_path = "./data_yegan/set_00/02_right.jpg"
img = cv.imread( img_path )

# resize image
scale_percent = 20  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()

kp = sift.detect(gray, None)

print( "keypoints: " , kp )

img = cv.drawKeypoints( gray, kp, img )

cv.imwrite('sift_keypoints_01.jpg',img)

cv.imshow( "SIFT B", img )

cv.waitKey(0)  
  
cv.destroyAllWindows() 

# end