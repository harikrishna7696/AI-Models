"""
HOG: Histogram of Oriented Gradients
How it works:
    1) Feature Descriptor: It simplifies the images by extracting the useful information and ignores rest.
        usually HOG converts image into (64*128) into a feature vector of length 3780 this will use for recognition
        and detection.
    2) Gradient Orientation Analysis: Hog computes the gradient magnitude and orientation
        *) gradient magnitude: The gradient magnitude at a specific pixel represents the strength or intensity of the
           change in pixel values (usually intensity or color),highlights the region with the significant intensity
           variations, such as degrees, corners, or textual boundaries.
        *) gradient orientation: a pixel indicates the directions of the intensity change. it provides the structure of
           the local image.
        gradient magnitude and orientation help us identify important features in an image, making them valuable for
        tasks like edge detection, object recognition, and image segmentation.
    3) Block Normalization: To account for variation in lightning and contrast HOG use the block-wise normalization
       each block(composed of multiple cells) is normalized to create the robust future representation, the normalized
       block concatenate to make the final feature vector.
    4) Object Detection: HOG features capture essential information about object shapes and appearances, when it fed into
       model perform well like image recognition and detection.
"""

import cv2

image = "C:/Users/smart/Desktop/running_image.jpg"

img = cv2.imread(image)
img = img /255.0

gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# Python Calculate gradient magnitude and direction ( in degrees )
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()