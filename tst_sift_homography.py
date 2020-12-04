# -*- coding: utf-8 -*-

import logging as log
log.basicConfig( format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import cv2, cv2 as cv
from matplotlib import pyplot as plt

from Common import *

common = Common()

img_path = "./data_yegan/set_01/_1018843.JPG"
img_path = "./data_yegan/set_00/01_left.jpg"

log.info( f"img_path = {img_path}" )

img_1 = cv2.imread(img_path, 0)
next_img_path = common.next_file( img_path, step=1 )
log.info( f"next_img_path = {next_img_path}" )

img_2 = cv2.imread(next_img_path, 0) # trainImage

# Initiate SIFT detector
sift = cv2.SIFT_create()

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

dim_square = max([len(img_1), len(img_1[0])]) * 0.3
dim_square = dim_square*dim_square

pts_src = []
pts_dst = []

goods = []
matchesMask = []

for i, (m1, m2) in enumerate(matches):
    valid = True

    ratio = 0.7
    #ratio = 0.9

    if valid and ( m1.distance > ratio * m2.distance ):
        valid = False
    pass

    if valid :
        pt1 = kp1[m1.queryIdx].pt
        pt2 = kp2[m1.trainIdx].pt

        dx = pt1[0] - pt2[0]
        dy = pt1[1] - pt2[1]

        if dx*dx + dy*dy > dim_square :
            valid = False
        pass

        if valid :
            pts_src.append( pt1 )
            pts_dst.append( pt2 )

            goods.append( [m1, m2] )
            matchesMask.append( [1, 0] )

            print(i, pt1, pt2 )

            if 1 and ( i % 5 == 0 ):
                # Draw pairs in purple, to make sure the result is ok
                cv2.circle(img_1, (int(pt1[0]), int(pt1[1])), 5, (255, 0, 0), -1)
                cv2.circle(img_2, (int(pt2[0]), int(pt2[1])), 5, (0, 0, 255), -1)
            pass
        pass
    pass
pass

h, status = cv2.findHomography( np.array(pts_src), np.array(pts_dst) )

img3 = cv2.warpPerspective(img_1, h, (img_2.shape[1], img_2.shape[0]))

#draw_params = dict(matchColor = (255, 0, 0), singlePointColor = (0, 0, 255), matchesMask = matchesMask, flags = 0)

#img3 = cv2.drawMatchesKnn(img_1, kp1, img_2, kp2, goods, None, **draw_params)

cv.imwrite('sift_homograpy.jpg', img3)

cv.imshow("SIFT HOMOGRAPHY", img3)

cv.waitKey(0)

cv.destroyAllWindows()