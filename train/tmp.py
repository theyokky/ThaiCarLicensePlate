import os
import numpy as np
import shutil

def copyImages(source_path, des_path, filenames):
     if not(os.path.isdir(des_path)):
          os.mkdir(des_path)
     for i in range(len(filenames)):
          source_filename = f"{source_path}/{filenames[i]}"
          des_filename = f"{des_path}/{filenames[i]}"
          print(f"\rcopy {source_filename} to {des_filename} ... {i+1}/{len(filenames)}", end=" "*10)
          shutil.copy(source_filename, des_filename)

data_path = "D:/ThaiCarLicensePlate/train/data_char/char_pre_processed_v11"
save_path = "D:/ThaiCarLicensePlate/train/data_char/char_pre_processed_v11_noise_normal_padding200x300"
     
# list class
classes = os.listdir(data_path)

# process each class
for ic in range(len(classes)):
     
     # list filenames
     filenames = np.array(os.listdir(f"{data_path}/{classes[ic]}"))

     # shuffle
     filenames = filenames[np.random.permutation(len(filenames))]
     
     # split data
     n_all = len(filenames)
     n_train = 5
     n_val = 2
     n_test = 3
     train_filenames = filenames[:n_train]
     val_filenames = filenames[n_train:(n_train+n_val)]
     test_filenames = filenames[(n_train+n_val):]
     
     # copy images
     copyImages(f"{data_path}/{classes[ic]}", f"{save_path}/train/{classes[ic]}", train_filenames)
     copyImages(f"{data_path}/{classes[ic]}", f"{save_path}/val/{classes[ic]}", val_filenames)
     copyImages(f"{data_path}/{classes[ic]}", f"{save_path}/test/{classes[ic]}", test_filenames)