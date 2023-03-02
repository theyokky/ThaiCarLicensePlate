import numpy as np
import tensorflow as tf
import cv2

class ThaiAlphabetClassifier():
    def __init__(self, model_filename='../models/best_model.h5'):
        self.model = tf.keras.models.load_model(model_filename)
        self.thai_alphabet_label = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", 
                                "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19", "c20", 
                                "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c28", "c29", "c30", 
                                "c31", "c32", "c33", "c34", "c35", "c36", "c37", "c38", "c39", "c40", 
                                "c41", "c42", "c43",
                                "n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"]
        """
        ['c0', 'c1', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c2', 
         'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c3', 
         'c30', 'c31', 'c32', 'c33', 'c34', 'c35', 'c36', 'c37', 'c38', 'c39', 'c4', 
         'c40', 'c41', 'c42', 'c43', 'c5', 'c6', 'c7', 'c8', 'c9', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9']
        """
    def classify(self,img_org):
        # img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (80,80))/255
        # img = np.dstack([img])
        # img = np.array([img])
        output = self.model.predict(img_org)
        return (output)