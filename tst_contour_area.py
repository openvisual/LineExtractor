# -*- coding:utf-8 -*-

import logging as log
log.basicConfig( format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)04d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=log.INFO )

import cv2, cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

img = cv.imread('./data/star.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, mode=1, method=2)

for i, contour in enumerate(contours):

    if 0 :
        M = cv.moments(contour)
        log.info(f"Moments = {M}")
    pass

    if 0 :
        # Straight Bounding Rectangle
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle( img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        log.info( f"x = {x}, y = {y}, w = {w}, h = {h}")
    pass

    if 0 :
        # minimum enclosing circle
        (x, y), radius = cv.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv.circle(img, center, radius, (255, 0, 0), 2)
    pass

    if 0 :
        # Fitting an Ellipse
        ellipse = cv.fitEllipse(contour)
        cv.ellipse(img, ellipse, (0, 255, 255), 2)
    pass

    if 1:
        # area and perimeter
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)

        log.info(f"[{(i + 1):03d}] area = {area}, perimeter = {perimeter}")
    pass

    # Rotated Rectangle
    min_rect = cv.minAreaRect(contour)
    min_box = cv.boxPoints( min_rect )
    min_box = np.int0(min_box)

    center = np.mean( min_box, axis=0 )

    text = [ ", ".join( item ) for item in min_box.astype(str) ]

    min_box_area = cv2.contourArea(min_box)

    area_width  = cv2.norm( min_box[0] - min_box[1], cv2.NORM_L2 )
    area_height = cv2.norm( min_box[1] - min_box[2], cv2.NORM_L2 )

    log.info( f"rotated_box = { text }, area = { min_box_area }, width = {area_width: .2f}, height = {area_height: .2f}")
    cv.drawContours(img, [ min_box ], 0, (0, 0, 255), 2, cv.LINE_AA)

    # Fitting a Line
    # (ax, ay) is a vector collinear to the line
    # (x0, y0) is a point on the line.
    [ax, ay, x0, y0] = cv.fitLine(contour, cv.DIST_L2, 0, 0.001, 0.001)

    log.info( f"fitLine: ax = {ax}, ay = {ay}, x0 = {x0}, y0 = {y0}")

    _, width = img.shape[:2]

    a = ay/ax

    d = sqrt( ax*ax + ay*ay )

    x1 = 0
    y1 = int( y0 + a * (x1 - x0) )

    x2 = width - 1
    y2 = int( y0 + a * (x2 - x0) )

    cv.line(img, (x1, y1), ( x2, y2), (255, 255, 0), 2, cv.LINE_AA)

    cv.circle(img, (int(x0), int(y0) ), 5, (255, 0, 255), 2, cv.LINE_AA)

    box_length = max( area_width, area_height )

    x1 = min_box[0][0]
    y1 = int(y0 + a * (x1 - x0))

    cv.circle(img, (int(x1), int(y1)), 5, (0, 255, 255), 2, cv.LINE_AA)

    x1 = min_box[1][0]
    y1 = int(y0 + a * (x1 - x0))

    cv.circle(img, (int(x1), int(y1)), 5, (255, 0, 0), 2, cv.LINE_AA)
pass

plt.imshow( img )
plt.show()

