import cv2 as cv
import os
import io
import re
#from transform import four_point_transform
import numpy as np
from google.cloud import vision_v1p3beta1 as vision


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ganesh/Desktop/Product_live/Incam_detection1/cardemo-3cbf87a35d8c.json"
cvNet = cv.dnn.readNetFromTensorflow('models/Outcam_model/frozen_inference_graph.pb', 'models/Outcam_model/outnum_model.pbtxt')
def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    # [START vision_python_migration_text_detection]
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    #print('Texts:')
    #print(texts[0].description.replace('\n',''))
    if len(texts)!=0:
    	return texts[0].description.replace('\n','')
    return None
    
    					
def number_detect(frame,file1):
	img = cv.imread(frame)
	rows = img.shape[0]
	cols = img.shape[1]
	cvNet.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
	cvOut = cvNet.forward()
	for detection in cvOut[0,0,:,:]:
		score = float(detection[2])
		if score > 0.3:
			left = int(detection[3] * cols)
			top = int(detection[4] * rows)
			right = int(detection[5] * cols)
			bottom = int( detection[6] * rows)
			#cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
			img4=img[top:bottom,left:right]
			cv.imwrite("images/num.jpg",img4)
			cv.imwrite("num2/num"+str(file1)+".jpg",img4)
			return detect_text('images/num.jpg')
	return None
			
					


