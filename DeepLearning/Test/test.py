import cv2
import numpy as np

# Read the image
image = cv2.imread('/Users/yshvrd/Downloads/Drive /LabPractical/DeepLearning/Test/Image.jpg')

height, width, channels = image.shape
mean = 0
sigma = 50
gaussian_noise = np.random.normal(mean, sigma, (height, width, channels))

noisy_image = image + gaussian_noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

display = np.hstack((image, noisy_image))
cv2.imshow('Original and Noisy Image', display)


# Quit
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
