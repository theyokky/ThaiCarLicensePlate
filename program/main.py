from LicensePlateDetector import LicensePlateDetector
import cv2
import os
import time

if __name__ == '__main__':
     
     get_license_plate = LicensePlateDetector()
     img = cv2.imread("img/car1.jpg")
     outputs = get_license_plate.getLicensePlate(img)
     # print(outputs)
     for i in range(len(outputs)):
          cv2.imwrite(f"img/{i}.jpg", outputs[i][2])
          print(float(outputs[i][1]))
          
