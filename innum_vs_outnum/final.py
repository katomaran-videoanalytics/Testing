import cv2
import numpy as np
from google.cloud import vision_v1p3beta1 as vision
import io
import re
import pandas as pd

def detect_text(path):
	client = vision.ImageAnnotatorClient()
	with io.open(path, 'rb') as image_file:
		content = image_file.read()
	image = vision.types.Image(content=content)
	response = client.text_detection(image=image)
	texts = response.text_annotations
	if len(texts)!=0:
		return re.sub('[^A-Za-z0-9]+', '', texts[0].description)
	return "None"

label=["GBB9032P","SLB9944P","SLP8772B","SKH5906G","SLZ300E","SLG9325A","SLU7747","SJY1928S","SLF4095B","SGK368Z","SJN7561J","SHA7822Z","SLG8804R","SMA2887K","SHC7520J","SLL8936S","SKC8287R","SKE5353S","SLM3669R","SJQ2807T","SLT3474Z","SLS7130U","SLN2539J","SKB9993J","SKQ9052X","SLQ6081U","SLJ8660Y","SLZ5663A","SLK7442L","SHB4493D","GBF5143M","SLM4222P","SLX8550H"]


for i in range(1,34):
	imagename="image"+str(i)+".jpg"
	gray_num1="gray_num1/num"+str(i)+".jpg"
	gray_num2="gray_num2/num"+str(i)+".jpg"
	print("processing = image"+str(i))
		
	gray_num1_text=detect_text(gray_num1)
	gray_num2_text=detect_text(gray_num2)
		
	raw_data={"Image_name":[imagename],"True_label":[label[i-1]],"incam_gray_image_label":[gray_num1_text],"outcam_gray_image_label":[gray_num2_text]}
	
	
	column_name = ['Image_name', 'True_label','incam_gray_image_label','outcam_gray_image_label']
	df = pd.DataFrame(raw_data, columns = column_name)
	df.to_csv('/home/ganesh/Desktop/testing/innum_vs_outnum/filter_results1.csv', mode='a', header=False)
	print("finised = image"+str(i))
	
	
	
