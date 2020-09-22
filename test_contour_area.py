# -*- coding:utf-8 -*-

import logging as log
log.basicConfig( format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import cv2, cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread('./data/star.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, mode=1, method=2)

for i, cnt in enumerate( contours ):
    M = cv.moments(cnt)
    0 and log.info( f"Moments = {M}" )

    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)

    log.info( f"[{(i+1):03d}] area = {area}, perimeter = {perimeter}")

    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle( img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    log.info( f"x = {x}, y = {y}, w = {w}, h = {h}")

    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img, [box], 0, (0, 0, 255), 2)
pass

plt.imshow( img )
plt.show()

