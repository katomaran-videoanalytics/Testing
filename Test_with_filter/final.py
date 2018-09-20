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

predected_label1=["GBB9032P","None","None","SKH59066","None","None","None","None","SLF409580","SGK368Z","SUN7561J","None","SLG8804R","SMA2887K","SHERUN","SLL8936S","None","SKE53535","SLM3669R","SJO28077","None","None","SLN2539)SLN2539","186668XS","None","None","None","None","None","None","None","SLM4222P","SLX8550H"]


predected_label2=["GBB9032P","None","SLP8772B","SKH5906G","SLZ300E","None","SLU7747E","None","None","SGK368Z","None","None","SLG8804R","None","None","SLL8936S","SKC287R","None","SLM3669R","None","SLT3474Z","SLS7130U","SLN2539J","SKB9993J","None","SLQ6081U","None","None","None","SHB4493D","None","SLM4222P","None"]
#print(len(predected_label1),len(predected_label2))
label=["GBB9032P","SLB9944P","SLP8772B","SKH5906G","SLZ300E","SLG9325A","SLU7747","SJY1928S","SLF4095B","SGK368Z","SJN7561J","SHA7822Z","SLG8804R","SMA2887K","SHC7520J","SLL8936S","SKC8287R","SKE5353S","SLM3669R","SJQ2807T","SLT3474Z","SLS7130U","SLN2539J","SKB9993J","SKQ9052X","SLQ6081U","SLJ8660Y","SLZ5663A","SLK7442L","SHB4493D","GBF5143M","SLM4222P","SLX8550H"]
#F = open("outnum.txt","w")
for i in range(1,34):
	imagename="image"+str(i)+".jpg"
	gray_path="gray_image/num"+str(i)+".jpg"
	denoised_image="denoised_image/num"+str(i)+".jpg"
	thr="threshold_withGaussian/num"+str(i)+".jpg"
	gr2de="grayimage2denoising/num"+str(i)+".jpg"
	gr2de2thr="grayimage2denoising2threshold/num"+str(i)+".jpg"
	print("processing = image"+str(i))
	
	
	gray_text=detect_text(gray_path)
	denoi_text=detect_text(denoised_image)
	thres=detect_text(thr)
	gray2denoi=detect_text(gr2de)
	gray2denois2thr=detect_text(gr2de2thr)
	
	
	raw_data={"Image_name":[imagename],"True_label":[label[i-1]],"Predected_label1":[predected_label1[i-1]],"Predected_label2":[predected_label2[i-1]],"Only_Gray_image_text":[gray_text],"Only_Denoising_text":[denoi_text],"Only_Adaptive_threshold_text":[thres],"Gray&Denoising_text":[gray2denoi],"Gray&Denoising&Adaptive_threshold_text":[gray2denois2thr]}
	column_name = ['Image_name', 'True_label','Predected_label1','Predected_label2','Only_Gray_image_text','Only_Denoising_text','Only_Adaptive_threshold_text','Gray&Denoising_text','Gray&Denoising&Adaptive_threshold_text']
	df = pd.DataFrame(raw_data, columns = column_name)
	df.to_csv('/home/ganesh/Desktop/test_with_filter/filter_results1.csv', mode='a', header=False)
	
	
	
	#F.write("image"+str(i)+".jpg"+" = "+label[i-1]+" = "+gray_text+" = "+denoi_text+" = "+thres+" = "+gray2denoi+" = "+gray2denois2thr+"\n")
	print("finised = image"+str(i))
	
	'''print(gray_text)
	print(denoi_text)
	print(thres)
	print(gray2denoi)
	print(gray2denois2thr)'''
	
	
	
