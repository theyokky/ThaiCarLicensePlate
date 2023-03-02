import numpy as np
from os import walk
import os
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # Astb412_400_30_22_161 , Astb312_300_30_18_161
# img = cv2.imread("Astb412_400_30_22_161.bmp")
# old_image_height, old_image_width, channels = img.shape
# print("img =         " ,old_image_height, old_image_width, channels)

# # รีไซส์ภาพด้วยบัญญัติไตรยางศ์ 
# # new_width = int((old_image_width/old_image_height)*200)
# # img_resize = cv2.resize(img, (new_width,200))
# # height, width, channels = img_resize.shape
# # print("img_resize = " ,height, width, channels)

# # convert color black and white
# img_invert = cv2.bitwise_not(img)
# cv2.imshow("inverted", img_invert)

# # padding
# new_image_width = 400
# new_image_height = 400
# color = (0,0,0)
# img_padding = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
# # compute center offset
# x_center = (new_image_width - old_image_width) // 2
# y_center = (new_image_height - old_image_height) // 2
# # copy img image into center of img_padding image
# img_padding[y_center:y_center+old_image_height, 
#        x_center:x_center+old_image_width] = img_invert

# # โชว์รูป
# cv2.imshow("img", img)
# # cv2.imshow("img_resize", img_resize)
# cv2.imshow("img_padding", img_padding)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 

# ------------------------------------------------------------


model_filename='best_model.h5'
model = tf.keras.models.load_model(model_filename)
body_shape_label = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", 
                     "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19", "c20", 
                     "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c28", "c29", "c30", 
                     "c31", "c32", "c33", "c34", "c35", "c36", "c37", "c38", "c39", "c40", 
                     "c41", "c42", "c43",
                     "n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"]

img_test = cv2.imread('Astb412_400_30_22_161.bmp')
img = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (150,150))/255
img = np.dstack([img])
img = np.array([img])
output = model.predict(img)
print(output)