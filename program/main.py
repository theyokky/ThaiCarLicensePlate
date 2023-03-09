from LicensePlateDetector import LicensePlateDetector
from CharacterDetector import CharacterDetector
from ThaiCharacterClassifier import ThaiAlphabetClassifier
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

class imgInformation():
     def __init__(self, name=""):
          self.id = None
          self.lcp_box = []
          self.lcp_img = []
          self.character_found = []
          self.character_box = []
          self.character_img = []
          self.character_str = []
     
     def writeOnFrame(self, font_path, font_size, frame, frame_result_path):
          print(len(self.character_found))
          print(len(self.character_box))
          print(len(self.character_img))
          print(len(self.character_str))
          for i in range(len(self.lcp_box)):
               text = ""
               x, y, w, h = self.lcp_box[i]
               cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0) , 1) 
               if self.character_found[i] != False:
                    for j in range(len(self.character_str[i])):
                         text += self.character_str[i][j]
                    font = ImageFont.truetype(font_path, font_size)
                    img_pil = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((x, y-32),  text, font = font, fill = (0,255,0))
                    frame = np.array(img_pil)
          cv2.imwrite(frame_result_path , frame)
          
     def predict(self, frame):
          thai_char_label = ['ก', 'ข', 'ซ', 'ณ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ฃ', 
                                    'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ค', 
                                    'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ฅ', 
                                    'ห', 'ฬ', 'อ', 'ฮ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 
                                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
          character_detector = CharacterDetector()
          license_plate_detector = LicensePlateDetector()
          char_clssifier = ThaiAlphabetClassifier()
          
          # lcp_img , lcp_box
          plates = license_plate_detector.getLicensePlate(frame)
          for i in range(len(plates)):
               self.lcp_img.append(plates[i][2])
               self.lcp_box.append(plates[i][0])
          
          # character_img , character_box
          for i in range(len(self.lcp_img)):
               char_list = []
               char_boxs = []
               img_tmp = self.lcp_img[i]
               chars = character_detector.getCharacter(img_tmp)
               if len(chars) > 0:
                    self.character_found.append(True)
                    for j in range(len(chars)):
                         char_list.append(chars[j][2]) 
                         char_boxs.append(chars[j][0])   
               else:
                    self.character_found.append(False)
                    char_list.append([]) 
                    char_boxs.append([])   
               self.character_img.append(char_list)
               self.character_box.append(char_boxs)
               
          #### ไว้ดูตัวอักษรที่ตัดมาได้เฉยๆ
          # for i in range(len(char_imgs)):
          #      cv2.imwrite(f"img/{i}_lcp.jpg" , char_imgs[i])
          
          # classify character
          for i in range(len(self.character_img)):
               char_list = []
               if len(self.character_img[i]) > 0 and self.character_img[i][0] != []:
                    for j in range(len(self.character_img[i])):
                         char_prob = char_clssifier.classify(self.character_img[i][j])
                         char_list.append(thai_char_label[np.argmax(char_prob)])
                    # เรียงตัวอักษร เพราะตอนแรก yolo ดีเท็คตัวอักษรมาให้แบบไม่เรียงกัน
                    list_tmp = np.array(self.character_box[i])
                    if len(list_tmp) != 0:
                         ind_sort = np.argsort(list_tmp[:,0])
                         char_list = [char_list[i] for i in ind_sort]
               self.character_str.append(char_list)
        
if __name__ == '__main__':
     
     # img_information = imgInformation(0)
     # img = cv2.imread("img/car2.jpg")
     
     # # predict
     # img_information.predict(img)
   
     # # write on frame
     # img_information.writeOnFrame("font/font_thai.ttf", 32, img, "img/result.jpg")
     
     cap = cv2.VideoCapture("img/IMG_6374.mp4")
     while(True):
          ret, frame = cap.read()
          # frame = cv2.resize(frame, (1000,1000))
          cv2.imshow('okkk', frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
               break
     cap.release()
     cv2.destroyAllWindows()
         

  
     
 