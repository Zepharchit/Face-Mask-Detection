import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Nadam
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import cv2
import argparse
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True)
args = vars(ap.parse_args())


label_dict = {
    0: 'Mask',
    1: 'Non_Mask',
}

path = args["image"]

model = load_model('./model/mask_model.h5')
face_net = cv2.dnn.readNet('deploy.prototxt.txt','res10_300x300_ssd_iter_140000.caffemodel')
file_1 = cv2.imread(path)
test_image = file_1.copy()

(h,w) = file_1.shape[:2]
blob = cv2.dnn.blobFromImage(file_1, 1.0, (300, 300),(104.0, 117.0, 123.0))
face_net.setInput(blob)
detections = face_net.forward()


for detection in range(0,detections.shape[2]):
    confidence = detections[0,0,detection,2]
    if confidence > 0.5:
        
        box = detections[0,0,detection,3:7] * np.array([w,h,w,h])
        (x_0,y_0,x_1,y_1) = box.astype('int')
        
        (x_0,y_0) = (max(0,x_0),max(0,y_0))
        (x_1,y_1) = (min(w-1,x_1),min(h-1,y_1))
        
        face = file_1[y_0:y_1,x_0:x_1]
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        face = cv2.resize(face,(331,331))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face,axis=0)
        
        pred = model.predict(face)[0]
        label = label_dict[np.argmax(pred)]
        colour = (0,255,0) if label == "Mask" else (0,0,255)
        
        cv2.putText(file_1, label, (x_0, y_0 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)
        cv2.rectangle(file_1, (x_0, y_0), (x_1, y_1), colour, 2)
        
    cv2.imshow("Output_",file_1)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
