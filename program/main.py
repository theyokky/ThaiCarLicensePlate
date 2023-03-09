from LicensePlateDetector import LicensePlateDetector
from CharacterDetector import CharacterDetector
from ThaiCharacterClassifier import ThaiAlphabetClassifier
import cv2
import os
import time
import numpy as np

class imgInformation():
     def __init__(self, name=""):
          self.id = None
          self.lcp_box = []
          self.lcp_img = []
          self.character_img = []
          self.character_str = []
        
if __name__ == '__main__':
     thai_char_label = ['ก', 'ข', 'ซ', 'ณ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ฃ', 
                                    'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ค', 
                                    'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ฅ', 
                                    'ห', 'ฬ', 'อ', 'ฮ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 
                                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
     character_detector = CharacterDetector()
     license_plate_detector = LicensePlateDetector()
     char_clssifier = ThaiAlphabetClassifier()
     
     img_information = imgInformation(0)
     img = cv2.imread("img/car2.jpg")
     
     # lcp_img , lcp_box
     plates = license_plate_detector.getLicensePlate(img)
     for i in range(len(plates)):
          img_information.lcp_img.append(plates[i][2])
          img_information.lcp_box.append(plates[i][0])
     
     # character_img
     char_imgs = []
     for i in range(len(img_information.lcp_img)):
          char_list = []
          img_tmp = img_information.lcp_img[i]
          chars = character_detector.getCharacter(img_tmp)
          for j in range(len(chars)):
               char_list.append(chars[j][2]) 
               char_imgs.append(chars[j][2])   
          img_information.character_img.append(char_list)   
          
     #### ไว้ดูตัวอักษรที่ตัดมาได้เฉยๆ
     for i in range(len(char_imgs)):
          cv2.imwrite(f"img/{i}_lcp.jpg" , char_imgs[i])
     
     # classify character
     for i in range(len(img_information.character_img)):
          for j in range(len(img_information.character_img[i])):
               char_prob = char_clssifier.classify(img_information.character_img[i][j])
               print(thai_char_label[np.argmax(char_prob)])
     
     # write information on frame
     for i in range(len(img_information.lcp_box)):
          x, y, w, h = img_information.lcp_box[i]
          cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0) , 1)    
     cv2.imwrite(f"img/result.jpg" , img)
