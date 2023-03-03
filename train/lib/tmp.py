import numpy as np
from os import walk
import os
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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