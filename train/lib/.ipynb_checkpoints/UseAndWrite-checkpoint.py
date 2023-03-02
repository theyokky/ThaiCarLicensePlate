import cv2
from ThaiAlphabetClassifier import ThaiAlphabetClassifier
import numpy as np
import os
import imutils

def padding(img, image_width=200, image_height=200):
     old_image_height, old_image_width , chanels = img.shape
     new_image_width = image_width
     new_image_height = image_height
     img_padding = np.full((new_image_height, new_image_width, chanels), (255,255,255), dtype=np.uint8)
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
     image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
     image_output = cv2.bitwise_not(image_output)
     image_output = thresholding(image_output, threshold)
     image_output = padding(image_output, padding_size, padding_size)
     return image_output

if __name__ == '__main__':
     
     data_path = "../data/test_pic2"
     labels = os.listdir(data_path)
     classes = ['c0', 'c1', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c2', 
               'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c3', 
               'c30', 'c31', 'c32', 'c33', 'c34', 'c35', 'c36', 'c37', 'c38', 'c39', 'c4', 
               'c40', 'c41', 'c42', 'c43', 'c5', 'c6', 'c7', 'c8', 'c9', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9']
     
     thai_alphabet_classifier = ThaiAlphabetClassifier()
     img_correct = 0
     img_all = 0
     for ic in range(len(labels)):
          filenames = os.listdir(f"{data_path}/{labels[ic]}")
          for ifile in range(len(filenames)):
               filename = f"{data_path}/{labels[ic]}/{filenames[ifile]}"
               print(filenames[ifile])
               img = cv2.imread(filename)
               # img = imutils.resize(img, height=200)
               img = cv2.resize(img, (100,200))
               
               img = padding(img, 800, 800)
               
               img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
               img = cv2.bitwise_not(img)

               img[img < 150] = 0
               # img[img >= 150] = 255
               img_copy = img.copy()
               
               # cv2.imshow("ok", img_copy)
               # cv2.waitKey(0)
               # cv2.destroyAllWindows() 
               
               img = cv2.resize(img , (80,80))/255
               img = np.dstack([img])
               img = np.array([img])
               
               new_path = f"../data/after2/padding_thr"
               
               predict = thai_alphabet_classifier.classify(img)
               predict = np.argmax(predict[0])
               print(labels[ic] , classes[predict])
               
               if labels[ic] == classes[predict]:
                    save_file = f"{new_path}/true/{filenames[ifile]}"
                    img_correct += 1
               else:
                    save_file = f"{new_path}/false/{filenames[ifile]}"
               img_all += 1
               cv2.imwrite(save_file, img_copy)
               
     print(str(img_correct) + " / " + str(img_all))
     
