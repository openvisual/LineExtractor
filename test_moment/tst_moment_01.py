# importing required libraries
import mahotas
import mahotas.demos
from pylab import gray, imshow, show
import numpy as np
import matplotlib.pyplot as plt

# loading iamge
img = mahotas.demos.load('lena')

# filtering image
img = img.max(2)

print( img.shape )

# radius
radius = 10

# computing zernike moments
value = mahotas.features.zernike_moments(img, radius)

# printing value

print( f"zernike moments : \n {value}" )

# showing image
imshow(img)
show()

