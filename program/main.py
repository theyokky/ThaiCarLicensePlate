from LicensePlateDetector import LicensePlateDetector
from CharacterDetector import CharacterDetector
import cv2
import os
import time

if __name__ == '__main__':
     
     # get_license_plate = LicensePlateDetector()
     # img = cv2.imread("img/car1.jpg")
     # outputs = get_license_plate.getLicensePlate(img)
     # # print(outputs)
     # for i in range(len(outputs)):
     #      cv2.imwrite(f"img/{i}.jpg", outputs[i][2])
          
     get_character = CharacterDetector()
     img = cv2.imread("img/lcp3.jpg")
     outputs = get_character.getCharacter(img)
     # print(outputs)
     for i in range(len(outputs)):
          cv2.imwrite(f"img/{i}_lcp.jpg", outputs[i][2])
