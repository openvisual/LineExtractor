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
    box = cv.boxPoints(cv.minAreaRect(contour))
    box = np.int0(box)

    text = [ ", ".join( item ) for item in box.astype(str) ]

    log.info( f"box = { text }" )
    cv.drawContours(img, [ box ], 0, (0, 0, 255), 2)

    # Fitting a Line
    height, width = img.shape[:2]

    # (vx, vy, x0, y0),
    # (vx, vy) is a vector collinear to the line
    # (x0, y0) is a point on the line.
    [ax, ay, x0, y0] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)

    log.info(f"fitLine: ax = {ax}, ay = {ay}, x0 = {x0}, y0 = {y0}")

    a = ay/ax

    b = y0 - a * x0
    y2 = y0 + a * (width - 1 - x0)

    #cv.line(img, ( int( x ), int( y ) ), ( int( vx ), int( vy ) ), (255, 255, 0), 2)
    cv.line(img, (0, int(b)), (width - 1, int(y2)), (255, 255, 0), 2)
pass

plt.imshow( img )
plt.show()

