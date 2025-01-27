# -*- coding: utf-8 -*-

import numpy as np
import cv2 , cv2 as cv
from matplotlib import pyplot as plt

from Common import *

common = Common()

img_path = "./data_yegan/set_01/_1018864.JPG"
img1 = cv2.imread( img_path, 0)
next_img_path = common.next_file( img_path )
img2 = cv2.imread( next_img_path , 0)

# Initiate ORB detector
orb = cv2.ORB_create()

# find the key points and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x: x.distance)

# Draw first 10 matches.
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[::] , None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3)
plt.show()