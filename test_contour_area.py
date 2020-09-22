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

    if 0 :
        M = cv.moments(cnt)
        log.info(f"Moments = {M}")
    pass

    if 0 :
        # Straight Bounding Rectangle
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle( img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        log.info( f"x = {x}, y = {y}, w = {w}, h = {h}")
    pass

    if 0 :
        # minimum enclosing circle
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv.circle(img, center, radius, (255, 0, 0), 2)
    pass

    if 0 :
        # Fitting an Ellipse
        ellipse = cv.fitEllipse(cnt)
        cv.ellipse(img, ellipse, (0, 255, 255), 2)
    pass

    if 1:
        # area and perimeter
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)

        log.info(f"[{(i + 1):03d}] area = {area}, perimeter = {perimeter}")
    pass

    # Rotated Rectangle
    box = cv.boxPoints( cv.minAreaRect(cnt) )
    box = np.int0(box)

    log.info( f"box = {box}")
    cv.drawContours(img, [ box ], 0, (0, 0, 255), 2)

    # Fitting a Line
    height, width = img.shape[:2]
    # -- (x,y) = (x0,y0) + t*(vx,vy)
    [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((width - x) * vy / vx) + y)
    #cv.line(img, ( int( x ), int( y ) ), ( int( vx ), int( vy ) ), (255, 255, 0), 2)
    cv.line(img, (width - 1, righty), (0, lefty), (255, 255, 0), 2)
pass

plt.imshow( img )
plt.show()

