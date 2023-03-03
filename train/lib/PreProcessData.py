from gc import callbacks
from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
from os import walk
import os
import cv2
import imutils

def makeNoise(img, size_pixel=30, color=[0,0,0], size_noise=11):
     img_height, img_width, img_chanels = img.shape
     img_noise = img.copy()
     
     random_pixel = np.random.randint(img_height-1, size=(2, size_pixel))
     for i in range(len(random_pixel[0])):
          if random_pixel[0][i] + size_noise < img_height-1 and random_pixel[1][i] + size_noise < img_height-1:
               img_noise[random_pixel[0][i]:random_pixel[0][i]+size_noise , random_pixel[1][i]:random_pixel[1][i]+size_noise] = color           
     return img_noise

def padding(img, image_width=200, image_height=200):
     old_image_height, old_image_width , channels  = img.shape
     new_image_width = image_width
     new_image_height = image_height
     img_padding = np.full((new_image_height, new_image_width , 3 ), 0, dtype=np.uint8)
     x_center = (new_image_width - old_image_width) // 2
     y_center = (new_image_height - old_image_height) // 2
     img_padding[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = img
     return img_padding

path = "data_char/char_train"
path_write = "data_char/char_pre_processed_v11"

for root, dirs, files in os.walk(path):
	for file_name in files:
		if(file_name.endswith(".bmp")):
			# print(os.path.join(root))
			str_folder = os.path.join(root).split("\\")[6]
			print(str_folder)
			# name_img = os.path.join(root,file).split("\\")[7]
			img = cv2.imread(os.path.join(root,file_name))
			
			if img.shape[0] > 30 and img.shape[1] > 30 and str_folder != 'n1':
				img = cv2.resize(img, (80,80))
				image_height, image_width, channels = img.shape

				# Normal pic
				img_invert = cv2.bitwise_not(img)
				img_invert_padding = padding(img_invert, 200 , 300)
				cv2.imwrite(f"{path_write}/{str_folder}/{file_name}", img_invert_padding)
    
				# Noise pic
				img_noise = makeNoise(img_invert)
				img_noise_padding = padding(img_noise , 200 , 300)
				file_noise = file_name.split(".")
				file_noise_name = file_noise[0] + "_noise." + file_noise[1]
				cv2.imwrite(f"{path_write}/{str_folder}/{file_noise_name}", img_noise_padding)
    
				# # many padding
				# file_ = file_name.split(".")
				# img_invert = cv2.bitwise_not(img)
				# cv2.imwrite(f"{path_write}/{str_folder}/{file_name}" , img_invert)
				# img_noise = makeNoise(img_invert)
				# file_name = f"{file_[0]}_n.{file_[1]}"
				# cv2.imwrite(f"{path_write}/{str_folder}/{file_name}" , img_noise)

				# for height in range(150, 451, 100):
				# 	for width in range(150, 451, 100):
						
				# 		if height == width :
      
				# 			img_padding = padding(img_invert , height , width)
				# 			file_name = f"{file_[0]}_padding{height}x{width}.{file_[1]}"   
				# 			cv2.imwrite(f"{path_write}/{str_folder}/{file_name}" , img_padding)

				# 			img_noise = makeNoise(img_invert)
				# 			img_noise_padding = padding(img_noise , height , width)
				# 			file_name = f"{file_[0]}_noise_padding{height}x{width}.{file_[1]}"
				# 			cv2.imwrite(f"{path_write}/{str_folder}/{file_name}" , img_noise_padding)
			
			if img.shape[0] > 20 and img.shape[1] > 15 and str_folder == 'n1':
				img = cv2.resize(img, (80,80))
				image_height, image_width, channels = img.shape
				
				# Normal pic
				img_invert = cv2.bitwise_not(img)
				img_invert_padding = padding(img_invert, 200 , 300)
				cv2.imwrite(f"{path_write}/{str_folder}/{file_name}", img_invert_padding)
    
				# Noise pic
				img_noise = makeNoise(img_invert)
				img_noise_padding = padding(img_noise , 200 , 300)
				file_noise = file_name.split(".")
				file_noise_name = file_noise[0] + "_noise." + file_noise[1]
				cv2.imwrite(f"{path_write}/{str_folder}/{file_noise_name}", img_noise_padding)
    
						