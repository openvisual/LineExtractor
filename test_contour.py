# -*- coding:utf-8 -*-

import cv2
import numpy as np

# Let's load a simple image with 3 black squares
img_path = "../data_yegan/_1018843.JPG"

image = cv2.imread( img_path )

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 30, 200)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
#mode = cv2.RETR_LIST
mode = cv2.RETR_EXTERNAL
(a,contours,b) = cv2.findContours(edged, cv2.mode, cv2.CHAIN_APPROX_NONE)

cv2.imshow('Canny Edges After Contouring', edged)

print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

