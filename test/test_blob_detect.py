# -*- coding: utf-8 -*-

import numpy as np
import cv2 , cv2 as cv
from matplotlib import pyplot as plt

from Common import *

common = Common()

img_path = "./data_yegan/set_01/_1018864.JPG"
image = cv2.imread( img_path, 0)          # queryImage
next_img_path = common.next_file( img_path )
img2 = cv2.imread( next_img_path , 0) # trainImage

# Load image
#image = cv2.imread( './data/BlobTest.jpg', 0)

# Set our filtering parameters
# Initialize parameter settiing using cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
#params.filterByArea = True
#params.minArea = 100

# Set Circularity filtering parameters
params.filterByCircularity = False
#params.minCircularity = 0.9

# Set Convexity filtering parameters
#params.filterByConvexity = True
#params.minConvexity = 0.2

# Set inertia filtering parameters
#params.filterByInertia = True
#params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(image)

# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

# Show blobs
plt.imshow( blobs )
plt.show()
