# coding: utf-8

from skimage import data
from skimage import io
#from skimage.filters import try_all_threshold
from my_skimage.thresholding import try_all_threshold
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

image = data.text()
image = data.page()
image = data.camera()
image = data.stereo_motorcycle()[0]

image = data.astronaut()

path = "./data_yegan/set_01/_1018843.JPG"
path = "./data_yegan/set_05/DJI_0004.JPG"
path = "./data_yegan/set_04/P1010015.JPG"

image = io.imread( path )

image = rgb2gray(image)

fig, ax = try_all_threshold(image, figsize=(15, 9), verbose=False)
fig.tight_layout()

plt.show()