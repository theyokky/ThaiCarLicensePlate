import cv2
from ThaiAlphabetClassifier import ThaiAlphabetClassifier
import numpy as np
import os
import imutils

def padding(img, image_width=250, image_height=250):
     old_image_height, old_image_width = img.shape
     new_image_width = image_width
     new_image_height = image_height
     img_padding = np.zeros((new_image_height, new_image_width), dtype=np.uint8)
     x_center = (new_image_width - old_image_width) // 2
     y_center = (new_image_height - old_image_height) // 2
     img_padding[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = img
     return img_padding

def thresholding(img, threshold=150):
     img_output = img.copy()
     img_output[img_output < threshold] = 0
     return img_output

def processImage(img, threshold=150, padding_size=250):
     image_output = img.copy()
     image_output = imutils.resize(image_output, height=50)
     image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
     image_output = cv2.bitwise_not(image_output)
     image_output = thresholding(image_output, threshold)
     image_output = padding(image_output, padding_size, padding_size)
     return image_output

if __name__ == '__main__':
     
     img = cv2.imread("../data/test_pic2/c15/IMG_4657(3).jpg")
     img = processImage(img, threshold=150, padding_size=130)
     
     cv2.imshow("img", img)
     cv2.waitKey(0)
     cv2.destroyAllWindows() 
     
     img = cv2.resize(img , (80,80))/255
     img_tmp = img.copy()
     img = np.dstack([img])
     img = np.array([img])
     
     thai_alphabet_classifier = ThaiAlphabetClassifier()
     predict = thai_alphabet_classifier.classify(img)
     predict = np.argmax(predict[0])
     print(predict)

     
    
     
     