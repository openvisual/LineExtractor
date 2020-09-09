# -*- coding: utf-8 -*-

import numpy as np
import cv2 , cv2 as cv
from matplotlib import pyplot as plt

from Common import *

common = Common()

img_path = "./data_yegan/set_01/_1018864.JPG"
img1 = cv2.imread( img_path, 0)          # queryImage
next_img_path = common.next_file( img_path )
img2 = cv2.imread( next_img_path , 0) # trainImage

imgL = cv.imread( img_path, 0)
imgR = cv.imread( next_img_path, 0)
rate = 2
stereo = cv.StereoBM_create(numDisparities=16*rate, blockSize=5)
disparity = stereo.compute(imgL, imgR)

plt.imshow( imgL, 'gray')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

plt.imshow( imgR, 'gray')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

plt.imshow(disparity, 'gray')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()