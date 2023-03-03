# ThaiCarLicensePlate
โปรแกรมนี้จะทำการ detect ป้ายทะเบียนรถ จากนั้นแสดงเลขทะเบียนบนป้ายทะเบียนรถนั้นให้มีความแม่นยำมากที่สุด
 
### 6210450539 ญาณกร จารุเดชศิริ
### 6210451691 พันธุ์ธัช เสมสมญาต

------------------------------------------------------------------------------------------------------------------------

## โปรแกรมสำหรับส่วนใช้งานจริง (กำลังพัฒนา)
จะอยู่ในโฟลเดอร์ program โดยมีโครงสร้างดังนี้

![alt text](https://github.com/theyokky/ThaiCarLicensePlate/blob/main/img/note1.JPG?raw=true)

โดยแต่ละไฟล์และโฟลเดอร์ จะมีหน้าที่ดังนี้

- โฟลเดอร์ data จะเก็บไฟล์นามสกุล yaml เอาไว้สำหรับใช้กับโมเดล YOLOv5 ในตัวโปรแกรมของเรา
- โฟลเดอร์ img ใช้เก็บภาพสำหรับทดลองรันโปรแกรม
- โฟลเดอร์ models ใช้เก็บโมเดลและโปรแกรมสำหรับการใช้งาน YOLOv5 เอาไว้
- โฟลเดอร์ utils จะเก็บไฟล์ต่างๆที่จำเป็นสำหรับการใช้งาน YOLOv5 เอาไว้
- โฟลเดอร์ weights จะเก็บ weights และ models สำหรับใช้ในตัวโปรแกรมของเราเอาไว้
- ไฟล์ CharacterDetector.py ใช้สำหรับ detect ป้ายทะเบียนรถบนรูปภาพ โดยจะรับ input เป็นรูปภาพ จากนั้นให้ output เป็นอาเรย์รูปภาพป้ายทะเบียนรถที่ detect ได้ในรูปภาพนั้นๆ
- ไฟล์ LicensePlateDetector.py ใช้สำหรับ detect ตัวอักษร ตัวเลข และจังหวัดบนรูปภาพป้ายทะเบียนรถ โดยจะรับ input เป็นรูปภาพ จากนั้นให้ output เป็นอาเรย์รูปภาพตัวอักษร ตัวเลข และจังหวัดที่ detect ได้ในรูปภาพนั้นๆ
- ไฟล์ ThaiCharacterClassifier.py ใช้สำหรับ clssify ตัวอักษรไทย โดยจะรับ input เป็นรูปภาพ จากนั้นให้ output เป็นความเป็นไปได้สำหรับตัวอักษรหรือตัวเลขในภาพนั้นๆ คลาสไหนมีค่ามากที่สุดจะตัดสินใจเป็นคลาสนั้น เช่น c0 คือ ก
- ไฟล์ export.py เป็นไฟล์ที่เอาไว้ใช้เรียกใช้งานโปรแกรม YOLOv5
- ไฟล์ main.py เป็นโปรแกรมหลักในการใช้รันโปรแกรม

------------------------------------------------------------------------------------------------------------------------

## ส่วนของการ Train โมเดล
จะอยู่ในโฟลเดอร์ train โดยมีโครงสร้างดังนี้

![alt text](https://github.com/theyokky/ThaiCarLicensePlate/blob/main/img/note2.JPG?raw=true)
      
โดยแต่ละไฟล์และโฟลเดอร์ จะมีหน้าที่ดังนี้

- โฟลเดอร์ data_char จะเก็บโฟลเดอร์ดาต้ารูปภาพสำหรับเทรนโมเดล CharacterDetector เอาไว้ 
- โฟลเดอร์ lib จะเก็บไฟล์นามสกุล py และ ipynb เอาไว้ โดยในโฟลเดอร์จะประกอบด้วยไฟล์ที่มีลำดับขั้นตอนดังนี้

     1. การเทรน LicensePlateDetector Model
     
          เนื่องจากเป็นการใช้โมเดล YOLOv5 มาทำ Transfer Learning จึงแนะนำให้เทรนไฟล์ *TrainYolov5_CarLicensePlate.ipynb* บน colab 
          โดยทางผู้จัดทำจะเชื่อม colab กับ google drive ที่เก็บดาต้าที่ใช้สำหรับเทรนเอาไว้
          สามารถเข้าไปดูดาต้าและดาวน์โหลดเพื่อเทรนได้ที่ https://drive.google.com/drive/folders/11sDgoP3YTBUPNkhVMoB0-yObhQ5_hFwC

     2. การเทรน CharacterDetector Model

          เนื่องจากเป็นการใช้โมเดล YOLOv5 มาทำ Transfer Learning จึงแนะนำให้เทรนไฟล์ *TrainYolov5LicensePlate.ipynb* บน colab 
          โดยทางผู้จัดทำจะเชื่อม colab กับ google drive ที่เก็บดาต้าที่ใช้สำหรับเทรนเอาไว้
          สามารถเข้าไปดูดาต้าและดาวน์โหลดเพื่อเทรนได้ที่ https://drive.google.com/drive/folders/1H7Dawsy4CVbpmp-LRLVAiB8Vhxbqdb6v

     3. การเทรน ThaiCharacterClassifier Model

          3.1 *PreProcessData.py* จะทำหน้าที่จัดการดาต้าจากโฟลเดอร์ char_train มารวบรวมให้เป็นโฟลเดอร์ char_pre_processed_v11 พร้อมทั้งจัดการเพิ่ม padding และ noise หลังจากการทดลองแล้วการทำ padding ขนาด 200x300 และเพิ่ม noise แล้วจะให้ผลลัพธ์ออกมาดีที่สุดคือ accuracy 83%

          3.2 *MakeTrainTestValData.py* จะทำหน้าที่นำดาต้ารูปภาพจากโฟลเดอร์ char_pre_processed_v11 มารวบรวมเป็นโฟลเดอร์ char_pre_processed_v11_noise_normal_padding200x300 ที่มีการแบ่งดาต้าเป็น Train , Test และ Validation ให้พร้อมสำหรับการเข้าเทรนโมเดล

          3.3 *TrainChar.py* จะเป็นไฟล์ที่ทำการเทรนโมเดล ThaiCharacterClassifier ด้วยการใช้ไลบรารี Tensorflow และ Keras ผลลัพธ์จะออกมาเป็นไฟล์นามสกุล h5 ในโฟลเดอร์ models

          และเนื่องจาก Github นั้นไม่สามารถอัพโหลดไฟล์หรือโฟลเดอร์ที่มีขนาดใหญ่มากได้ เราจึงอัพโหลดภาพดาต้าตัวอย่างลงในโฟลเดอร์ char_train โดยดาต้าจริงสามารถสามารถเข้าไปดูและดาวน์โหลดเพื่อเทรนได้ที่ 

- โฟลเดอร์ models จะเก็บโมเดลที่ชื่อว่า best_model11_noise_normal_padding200x300.h5 ที่ได้จากการเทรน ThaiCharacterClassifier Model เอาไว้

------------------------------------------------------------------------------------------------------------------------

## วิธีการใช้งาน (กำลังพัฒนา)

โดยจะมีวิธีการทำงานหลักรวมๆ ดังรูปนี้

     
1. รันไฟล์ main.py ในโฟลเดอร์ program 

โดยตอนนี้กำลังอยู่ในขั้นตอนทดลองใช้งานคลาสและฟังก์ชัน LicensePlateDetector และ CharacterDetector
คลาสและฟังก์ชัน ThaiCharacterClassifier กำลังพัฒนาพร้อมๆไปกับการรวมโปรแกรม




