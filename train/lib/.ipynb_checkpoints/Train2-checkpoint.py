import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
def loadImages(data_path, size=(50,50)):
     x = []
     t = []
     classes = os.listdir(data_path)
     for ic in range(len(classes)):
          filenames = os.listdir(f"{data_path}/{classes[ic]}")
          for ifile in range(len(filenames)):
               filename = f"{data_path}/{classes[ic]}/{filenames[ifile]}"
               print(f"\rload {filename} ... {ifile+1}/{len(filenames)}",end=" "*10)
               image = cv2.imread(filename)
               image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
               image = cv2.resize(image, size)/255
               image = np.dstack([image])
               x.append(image)
               t.append(ic)
     x = np.array(x, dtype=np.float32)
     t = np.array(t, dtype=np.float32)
     return x, t

if __name__ == '__main__':
     
     # set parameters
     data_path = "../data/char_pre_processed_v2_2"
     save_data = "data_loaded3.mat"
     model_filename = "../models/best_model3.h5"
     
     # load images
     x_train, t_train = loadImages(f"{data_path}/train")
     x_val, t_val = loadImages(f"{data_path}/val")
     x_test, t_test = loadImages(f"{data_path}/test")
     print(f"train size",x_train.shape, t_train.shape)
     print(f"val size",x_val.shape, t_val.shape)
     print(f"test size",x_test.shape, t_test.shape)
     
     # model structure
     model = tf.keras.Sequential([
       tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu',input_shape=(60, 60, 1)),
       tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
       tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
       tf.keras.layers.MaxPooling2D(pool_size=(3,3)),
       tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
       tf.keras.layers.MaxPooling2D(pool_size=(5,5)),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(128),
       tf.keras.layers.Dense(54 ,activation='softmax')
     ])
     model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
     mc = ModelCheckpoint(model_filename, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

     # training
     model.fit(x_train, t_train, validation_data=(x_val, t_val), callbacks=[es,mc], epochs=100000)

