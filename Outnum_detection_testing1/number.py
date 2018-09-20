import cv2 as cv
import os
import io
import re
import numpy as np
from google.cloud import storage
from google.cloud import vision
from google.protobuf import json_format
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ganesh/Desktop/Product_live/Incam_detection1/cardemo-3cbf87a35d8c.json"
cvNet = cv.dnn.readNetFromTensorflow('models/Number_plate_Model/frozen_inference_graph.pb', 'models/Number_plate_Model/NuPlate_model.pbtxt')

def denoising(path):
	img = cv.imread('Results/gray_image.png')
	b,g,r = cv.split(img)           # get b,g,r
	rgb_img = cv.merge([r,g,b])     # switch it to rgb

	# Denoising
	dst = cv.fastNlMeansDenoising(img,None,10,7,21) #For RGB-cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

	b,g,r = cv.split(dst)           # get b,g,r
	rgb_dst = cv.merge([r,g,b])     # switch it to rgb
	cv.imwrite('Results/denoising.png',rgb_dst)
	return rgb_dst


def conv_grayscale(path):
	image = cv.imread('Results/final.jpg')
	gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	cv.imwrite('Results/gray_image.png',gray_image)
	return denoising('Results/gray_image.png')
	
	
	  
def increse_contrast(path):
	img = cv.imread(path)

	lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)

	l, a, b = cv.split(lab)

	clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	cl = clahe.apply(l)

	limg = cv.merge((cl,a,b))

	final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
	cv.imwrite('Results/final.jpg',final)
	return conv_grayscale('Results/final.jpg')
	
def detect_document(path):
	client = vision.ImageAnnotatorClient()

	# [START vision_python_migration_document_text_detection]
	with io.open(path, 'rb') as image_file:
		content = image_file.read()

	image = vision.types.Image(content=content)

	response = client.document_text_detection(image=image)
	stri=''
	sav=0
	count=0
	for page in response.full_text_annotation.pages:
		for block in page.blocks:
			for paragraph in block.paragraphs:
				for word in paragraph.words:
					word_text = ''.join([symbol.text for symbol in word.symbols])
					stri+=word_text

	return stri
					
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
			cont=increse_contrast("images/num.jpg")
			cv.imwrite('gimp.png',cont)
			return detect_document('gimp.png')
	return None

			
					


