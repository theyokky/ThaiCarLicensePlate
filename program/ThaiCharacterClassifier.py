import numpy as np
import tensorflow as tf
import cv2
import imutils

def padding(img, image_width=200, image_height=200):
    old_image_height, old_image_width  = img.shape
    new_image_width = image_width
    new_image_height = image_height
    img_padding = np.full((new_image_height, new_image_width ), 0, dtype=np.uint8)
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2
    img_padding[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = img
    return img_padding

class ThaiAlphabetClassifier():
    def __init__(self, model_filename='weights/best_model11_noise_normal_padding200x300.h5'):
        self.model = tf.keras.models.load_model(model_filename)
        self.thai_alphabet_label = ['ก', 'ข', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ฃ', 
                                    'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ค', 
                                    'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ฅ', 
                                    'ห', 'ฬ', 'อ', 'ฮ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 
                                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                                
    def classify(self,img_org):
        # img = np.array(img_org)
        img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY) 
        img = cv2.bitwise_not(img)
        
        ret,img = cv2.threshold(img,0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = imutils.resize(img, height=200)
        
        img = cv2.resize(img, (100,200))

        img = padding(img, 250 , 650)
        
        img[img <= 200] = 0
        # img[img > 100] = 255
        
        # cv2.imshow("ok", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        img = cv2.resize(img , (80,80))
        img = img/255
        img = np.dstack([img])
        img = np.array([img] ,dtype=np.float32)
        
        output = self.model.predict(img)
        return (output)