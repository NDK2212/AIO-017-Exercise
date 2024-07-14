import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
the_img = 'MODULE2\WEEK1_06072024\dog.jpeg'
# 12
img = mpimg.imread(the_img)
print(img.shape)
gray_img_01 = (img.max(axis=2) + img.min(axis=2))/2
print(gray_img_01[0, 0])
plt.imshow(gray_img_01, cmap='gray')
plt.axis('off')  # Hide the axis
plt.show()
# 13
img = mpimg.imread(the_img)
print(img.shape)
gray_img_02 = (np.sum(img, axis=2))/3
print(gray_img_02[0, 0])
plt.imshow(gray_img_02, cmap='gray')
plt.axis('off')  # Hide the axis
plt.show()
# 14
img = mpimg.imread(the_img)
print(img.shape)
gray_img_03 = (img[:, :, 0]*0.21 + img[:, :, 1]*0.72 + img[:, :, 2]*0.07)
print(gray_img_03[0, 0])
plt.imshow(gray_img_03, cmap='gray')
plt.axis('off')  # Hide the axis
plt.show()
