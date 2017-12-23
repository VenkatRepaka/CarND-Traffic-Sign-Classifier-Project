import cv2
import numpy as np

image = cv2.imread("jurassic-park-tour-jeep.jpg")
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyWindow("image")
rows, cols, ch = image.shape

pts1 = np.float32([[194,136],[470,152],[79,324],[574,272]])
pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

M = cv2.getPerspectiveTransform(pts1,pts2)

perspective_changed = cv2.warpPerspective(image,M,(300,300))
cv2.namedWindow('perspective_changed', cv2.WINDOW_NORMAL)
cv2.imshow("perspective_changed", perspective_changed)
cv2.waitKey(0)
cv2.destroyWindow("perspective_changed")

