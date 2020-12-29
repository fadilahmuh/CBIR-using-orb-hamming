import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.io.matlab.mio5_params import mat_struct

image1 = cv2.imread('image/2_225.jpg')  # queryImage
image2 = cv2.imread('image/2_224.jpg')  # trainImage
img1 = cv2.imread('image/2_225.jpg', 0) # queryImage
img2 = cv2.imread('image/2_224.jpg', 0) # trainImage

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches ,None, flags=2)
print(len(des1))
print('img',img1)
print('des',des1)
print(len(des2))
print(len(matches))
plt.imshow(image1),plt.show()
plt.imshow(image2),plt.show()