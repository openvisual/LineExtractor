# -*- coding:utf-8 -*-

import cv2
import matplotlib.pyplot as plt

# Let's load a simple image with 3 black squares
img_path = "./data_yegan/set_01/_1018843.JPG"
grayscale = cv2.imread( img_path, 0 )

img = grayscale

def threadhold_li( img ) :
    import numpy as np
    from skimage import filters

    def quantile_95(image):
        # you can use np.quantile(image, 0.95) if you have NumPy>=1.15
        return np.percentile(image, 95)
    pass

    iter_thresholds = []
    opt_threshold = filters.threshold_li( img, initial_guess=quantile_95,
                                          iter_callback=iter_thresholds.append)

    print(len(iter_thresholds), 'examined, optimum:', opt_threshold)

    data = img > opt_threshold

    return data
pass

fig, axes = plt.subplots( 1, 2 )

ax = axes[0]
ax.imshow(img, cmap='gray')
ax.set_title('image')
ax.set_axis_off()

bin = threadhold_li( img )
ax = axes[1]
ax.imshow( bin, cmap='gray')
ax.set_title('thresholded')
ax.set_axis_off()

#fig.tight_layout()

plt.show()