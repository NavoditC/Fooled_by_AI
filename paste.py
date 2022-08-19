import cv2
import math
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

def showPlot(image):
    plot = plt.imshow(image)
    plt.axis('off')
    plt.show()

# img_fullmoon = cv2.imread('C:\\Users\\Hp\\Desktop\\input_images\\yellow_full_moon.jpg')
# img = cv2.imread('C:\\Users\\Hp\\Desktop\\input_images\\sim_img_8_adv.PNG')

img_fullmoon = Image.open('C:\\Users\\Hp\\Desktop\\input_images\\yellow_full_moon.jpg')
img = Image.open('C:\\Users\\Hp\\Desktop\\input_images\\sim_img_8.PNG')

print("yes")

img_fullmoon = np.asarray(img_fullmoon)
img = np.asarray(img)

# cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
# cv2.imshow('Image', img)

showPlot(img)
showPlot(img_fullmoon)

# y = img.shape[0]
x = img.shape[1]

zeros = np.zeros((100, x, 3))
# dummy = img[:100, :, :]

img[:100, :, :] = zeros

# cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
# cv2.imshow('Image', img)

m = img_fullmoon.shape[0]

pivot_x = math.floor((x - m)/2)
pivot_y = 50 - math.floor(m/2)

# pivot = (pivot_x, pivot_y)
# print(pivot)

# dummy = img[pivot_y:pivot_y + m, pivot_x:pivot_x + m, :]
img[pivot_y:pivot_y + m, pivot_x:pivot_x + m, :] = img_fullmoon

# cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
# cv2.imshow('Image', img)

showPlot(img)

# k = cv2.waitKey()
# cv2.destroyAllWindows()
