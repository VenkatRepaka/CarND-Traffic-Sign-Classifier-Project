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

# Rotate image fixing a center
(h, w) = image.shape[:2]
center = (w/4, h/4)

M = cv2.getRotationMatrix2D(center, 20, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("rotated", rotated)
cv2.waitKey(0)
cv2.destroyWindow("rotated")

# Shift image

M = np.float32([[1,0,100],[0,1,50]])
(h, w) = image.shape[:2]
shifted = cv2.warpAffine(image, M, (w, h))

cv2.imshow('shifted', shifted)
cv2.waitKey(0)
cv2.destroyWindow("shifted")

# Affine transformation
rows, cols, ch = image.shape

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 200]])

M = cv2.getAffineTransform(pts1,pts2)
affined = cv2.warpAffine(image, M, (cols*2, rows*2))

# M = np.float32([[1,0,50],[0,1,100]])
# (h, w) = image.shape[:2]
# affined = cv2.warpAffine(affined, M, (w, h))
cv2.namedWindow('affined', cv2.WINDOW_NORMAL)
cv2.imshow("affined", affined)
cv2.waitKey(0)
cv2.destroyWindow("rotated")
