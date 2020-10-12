# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt

from Common import *

common = Common()

img_path = "./data_yegan/set_01/_1018864.JPG"
img_path = "./data_line/shapes_and_colors 01.png"

img1 = cv2.imread( img_path, 1)          # queryImage
gray1 = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY)

next_img_path = common.next_file( img_path )
img2 = cv2.imread( next_img_path , 1) # trainImage
gray2 = cv2.cvtColor( img2, cv2.COLOR_BGR2GRAY)


# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute( gray1, None)
kp2, des2 = sift.detectAndCompute( gray2, None)

img1 = cv2.drawKeypoints( gray1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2 = cv2.drawKeypoints( gray2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fig, axes = plt.subplots( nrows=1, ncols=2)
axes[0].imshow(img1)
axes[1].imshow(img2)
plt.show()

plt.imsave( 'c:/temp/img_1.png', img1)
plt.imsave( 'c:/temp/img_2.png', img2)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

max_dy = len( img1 ) * 0.3
max_dx = len( img1[0] ) * 0.3

# Apply ratio test
goods = []
for m, n in matches:
    valid = True
    if valid and m.distance > 0.7*n.distance:
        valid = False
    pass

    if valid:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[n.trainIdx].pt

        dx = abs( pt1[0] - pt2[0] )
        dy = abs( pt1[1] - pt2[1] )

        if dx > max_dx or dy > max_dy :
            valid = False
        pass
    pass

    if valid :
        goods.append([m])
    pass
pass

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, goods, None, flags=2)

plt.imshow(img3)
plt.show()

plt.imsave( 'c:/temp/img_3.png', img3)