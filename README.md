# ThaiCarLicensePlate
โปรแกรมนี้จะทำการ detect ป้ายทะเบียนรถ จากนั้นแสดงเลขทะเบียนบนป้ายทะเบียนรถนั้นให้มีความแม่นยำมากที่สุด
 
### 6210450539 ญาณกร จารุเดชศิริ
### 6210451691 พันธุ์ธัช เสมสมญาต

------------------------------------------------------------------------------------------------------------------------

## โปรแกรมสำหรับส่วนใช้งานจริง
จะอยู่ในโฟลเดอร์ program โดยมีโครงสร้างดังนี้
.
├── program
│   └── data
|       └── character.yaml
|       └── licensePlate.yaml
│   └── img 
│   └── models
│   └── utils
│   └── weights
│   └── CharacterDetector.py
│   └── LicensePlateDetector.py
│   └── ThaiCharacterClassifier.py
│   └── export.py
│   └── main.py













------------------------------------------------------------------------------------------------------------------------
## ส่วนของการ Train โมเดล
จะอยู่ในโฟลเดอร์ train โดยมีโครงสร้างดังนี้

.
├── train
│   └── data_char
|       └── char_pre_processed_v11
|       └── char_pre_processed_v11_noise_normal_padding200x300
|       └── char_train
│   └── lib 
│   └── models


      



