import cv2
import numpy as np

# Show original image
image = cv2.imread("jurassic-park-tour-jeep.jpg")
cv2.imshow("original", image)
cv2.waitKey(0)
cv2.destroyWindow("original")

# Shrink or expand image
ratio = 100.0/image.shape[1]
new_dim = (100, int(image.shape[0]*ratio))

resized = cv2.resize(image, new_dim, interpolation = cv2.INTER_AREA)
cv2.imshow("resized", resized)
cv2.waitKey(0)
cv2.destroyWindow("resized")