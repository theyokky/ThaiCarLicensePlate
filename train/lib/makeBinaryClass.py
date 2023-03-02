import numpy as np
from os import walk
import os
import cv2
import imutils
import time
import shutil

def copyImages(source_path, des_path, filenames):
     if not(os.path.isdir(des_path)):
          os.mkdir(des_path)
     for i in range(len(filenames)):
          source_filename = f"{source_path}/{filenames[i]}"
          des_filename = f"{des_path}/{filenames[i]}"
          print(f"\rcopy {source_filename} to {des_filename} ... {i+1}/{len(filenames)}", end=" "*10)
          shutil.copy(source_filename, des_filename)
          
if __name__ == '__main__':
     
     data_path = "D:/study/thai_alphabet/data/char_pre_processed_v11"
     save_path = "D:/study/thai_alphabet/data/char_pre_processed_n2"
     
     name_class_yes = "n2"
     name_class_no = "not_n2"
     
     # make dir
     if not(os.path.isdir(save_path)):
          os.mkdir(save_path)
     if not(os.path.isdir(f"{save_path}/{name_class_yes}")):
          os.mkdir(f"{save_path}/{name_class_yes}")
     if not(os.path.isdir(f"{save_path}/{name_class_no}")):
          os.mkdir(f"{save_path}/{name_class_no}")
          
     # list class : all : c0, c1, c2, ..... ,n0 ,n1 , .... ,n9
     classes = os.listdir(data_path)
     
     # process each class
     for ic in range(len(classes)):
          
          # list filenames
          filenames = os.listdir(f"{data_path}/{classes[ic]}")
          
          # copy images
          if classes[ic] == name_class_yes :
               copyImages(f"{data_path}/{classes[ic]}", f"{save_path}/{name_class_yes}", filenames)
          else:
               copyImages(f"{data_path}/{classes[ic]}", f"{save_path}/{name_class_no}", filenames)
               
          
          
          