# importing required libraries
import mahotas
import numpy as np
from pylab import gray, imshow, show
import os
import matplotlib.pyplot as plt

# loading iamge
img = mahotas.imread('dog_image.jpg')

# fltering image
img = img[:, :, 0]

# radius
radius = 10

# computing zernike moments
value = mahotas.features.zernike_moments(img, radius)

# printing value
print( f"zernike moments : \n {value}" )

# showing image
imshow(img)
show()
