from LicensePlateDetector import LicensePlateDetector
from CharacterDetector import CharacterDetector
from ThaiCharacterClassifier import ThaiAlphabetClassifier
import cv2
import os
import time
import numpy as np

class LCPInformation():
     def __init__(self, name="car"):
          self.id = None
          self.box = []
          self.lcp_img = []
          self.character_img = []
          self.character = []
        
if __name__ == '__main__':
     thai_char_label = ['ก', 'ข', 'ซ', 'ณ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ฃ', 
                                    'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ค', 
                                    'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ฅ', 
                                    'ห', 'ฬ', 'อ', 'ฮ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 
                                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
     
     character_detector = CharacterDetector()
     license_plate_detector = LicensePlateDetector()
     char_clssifier = ThaiAlphabetClassifier()
     
     lcr = LCPInformation(0)
     
     img = cv2.imread("img/car4.jpg")
     
     plates = license_plate_detector.getLicensePlate(img)
     
     # lcp_img , box
     for i in range(len(plates)):
          lcr.lcp_img.append(plates[i][2])
          lcr.box.append(plates[i][0])
          cv2.imwrite(f"img/{i}.jpg", plates[i][2])
     
     # character_img
     for i in range(len(lcr.lcp_img)):
          img = lcr.lcp_img[i]
          char_list = character_detector.getCharacter(img)
          
     for i in range(len(char_list)):
          lcr.character_img.append(char_list[i][2])
          cv2.imwrite(f"img/{i}_lcp.jpg", char_list[i][2])
          
     # classify character
     for i in range(len(char_list)):
          char_prob = char_clssifier.classify(char_list[i][2])
          print(thai_char_label[np.argmax(char_prob)])
          
     print()
          
     
