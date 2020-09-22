# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt

from Common import *

common = Common()

img_path = "./data_yegan/set_01/_1018843.JPG"
img_1 = cv2.imread(img_path, 0)          # queryImage
next_img_path = common.next_file( img_path )
img_2 = cv2.imread(next_img_path, 0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img_1, None)
kp2, des2 = sift.detectAndCompute(img_2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

#Ratio test
matches = flann.knnMatch(des1, des2, k=2)

dim = max([len(img_1), len(img_1[0])]) * 0.1
dim = dim*dim

goods = []
matchesMask = []

for i, (m1, m2) in enumerate(matches):
    valid = True

    if valid and ( m1.distance > 0.7 * m2.distance ):
        valid = False
    pass

    if valid :
        pt1 = kp1[m1.queryIdx].pt
        pt2 = kp2[m1.trainIdx].pt

        dx = pt1[0] - pt2[0]
        dy = pt1[1] - pt2[1]

        if 0 and dx*dx + dy*dy > dim :
            valid = False
        pass

        if valid :
            goods.append( [m1, m2] )
            matchesMask.append( [1,0] )
            ## Notice: How to get the index
            print(i, pt1, pt2 )
            if i % 5 == 0:
                ## Draw pairs in purple, to make sure the result is ok
                cv2.circle(img_1, (int(pt1[0]), int(pt1[1])), 5, (255, 0, 255), -1)
                cv2.circle(img_2, (int(pt2[0]), int(pt2[1])), 5, (255, 0, 255), -1)
            pass
        pass
    pass
pass

draw_params = dict(matchColor = (255, 0, 0), singlePointColor = (0, 0, 255),
                   matchesMask = matchesMask, flags = 0)

img3 = cv2.drawMatchesKnn(img_1, kp1, img_2, kp2, goods, None, **draw_params)

plt.imshow(img3)
plt.show()