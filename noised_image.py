import cv2
import numpy as np

# Show original image
image = cv2.imread("jurassic-park-tour-jeep.jpg")
# cv2.imshow("original", image)
# cv2.waitKey(0)
# cv2.destroyWindow("original")


# def add_gaussian_noise(image_in, noise_sigma):
#     temp_image = np.float64(np.copy(image_in))
#
#     h = temp_image.shape[0]
#     w = temp_image.shape[1]
#     noise = np.random.randn(h, w) * noise_sigma
#
#     noisy_image = np.zeros(temp_image.shape, np.float64)
#     if len(temp_image.shape) == 2:
#         noisy_image = temp_image + noise
#     else:
#         noisy_image[:,:,0] = temp_image[:,:,0] + noise
#         noisy_image[:,:,1] = temp_image[:,:,1] + noise
#         noisy_image[:,:,2] = temp_image[:,:,2] + noise
#
#     """
#     print('min,max = ', np.min(noisy_image), np.max(noisy_image))
#     print('type = ', type(noisy_image[0][0][0]))
#     """
#
#     return noisy_image
#
#
# noisy_image = add_gaussian_noise(image, 120)
# cv2.imshow("noisy_image", noisy_image)
# cv2.waitKey(0)
# cv2.destroyWindow("noisy_image")


# noisy_img = image + np.random.normal(0.,0.0005,(388,647,3)).astype(np.float32)
noisy_img = image + np.random.normal(0.,20,(388,647,3)).astype(np.uint8)
cv2.imshow("noisy_img", noisy_img)
cv2.waitKey(0)
cv2.destroyWindow("noisy_img")

row,col,ch= image.shape
mean = 0.0
var = 0.8
sigma = var**0.5
print(sigma)
gauss = np.array(image.shape)
gauss = np.random.normal(0.0,20,(row,col,ch))
print(gauss.shape)
gauss = gauss.reshape(row,col,ch)
print(gauss.shape)
noisy_img = image + gauss
print(noisy_img)
noisy_img = noisy_img.astype('uint8')
print(noisy_img)
cv2.imshow("noisy_img", noisy_img)
cv2.waitKey(0)
cv2.destroyWindow("noisy_img")
