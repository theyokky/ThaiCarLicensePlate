from ThaiAlphabetClassifier import ThaiAlphabetClassifier
import cv2
import numpy as np
import os

def loadImages(data_path, size=(80,80)):
     x = []
     t = []
     classes = os.listdir(data_path)
     for ic in range(len(classes)):
          filenames = os.listdir(f"{data_path}/{classes[ic]}")
          for ifile in range(len(filenames)):
               filename = f"{data_path}/{classes[ic]}/{filenames[ifile]}"
               print(f"\rload {filename} ... {ifile+1}/{len(filenames)}",end=" "*10)
               image = cv2.imread(filename)
               image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
               image = cv2.resize(image, size)/255
               image = np.dstack([image])
               x.append(image)
               t.append(ic)
     x = np.array(x, dtype=np.float32)
     t = np.array(t, dtype=np.float32)
     return x, t

def predict(x):
     thai_alphabet_classifier = ThaiAlphabetClassifier()
     predict_list = thai_alphabet_classifier.classify(x)
     y = []
     for i in range(len(predict_list)):
          ind = np.argmax(predict_list[i])
          y.append(ind)
     y = np.array(y)
     return y

def writeFile(y, t):
     matrix = []
     for i in range(54):
          mini_matrix = []
          for j in range(54):
               mini_matrix.append(0)
          matrix.append(mini_matrix)
          
     for i in range(len(y)):
          matrix[y[i]][int(t[i])] += 1
          
     matrix_new = []
     for i in range(len(matrix)):
          for j in range(len(matrix[i])):
               matrix_new.append(matrix[i][j])
               
     counter = 0
     file_ = open("data.csv","w+")
     for i in range(54):
          for j in range(54):
               file_.write(f"{matrix_new[counter]}")
               if j == 53:
                    file_.write("\n")
               else:
                    file_.write(",")
               counter += 1
     file_.close()

def writeFileConfutionMatrix():
     f = open("data.csv", "r")
     data = []
     for x in f:
          data.append(x.strip("\n").split(","))
     f.close()
     
     f = open("confusion.txt", "w")
     tp = 0
     fp = 0
     fn = 0
     for i in range(len(data)):
          for j in range(len(data[i])):
               if i == j:
                    tp += int(data[i][j])
               else:
                    fp += int(data[i][j])
          for j in range(len(data[i])):
               if i != j:
                    fn += int(data[j][i])
          
          f.write(f"Class {i} \n")
          f.write(f"tp = {tp} , fp = {fp} , fn = {fn} \n")
          f.write(f"Precision = {tp / (tp+fp)} \n")
          f.write(f"Recall = {tp / (tp+fn)} \n")
          f.write("\n")
          
          tp = 0
          fp = 0
          fn = 0
          
     f.close()

if __name__ == '__main__':
     data_path = "char_pre_processed_2"
     x_test, t_test = loadImages(f"{data_path}/test")
     # print(f"test size",x_test.shape, t_test.shape)
     
     y_test = predict(x_test)
     # print(len(y_test))
      
     writeFile(y_test, t_test)
  
     writeFileConfutionMatrix()
     
     
          
     
