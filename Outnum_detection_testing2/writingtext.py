import glob
import cv2
import os
import re
from number import number_detect
F = open("outnum.txt","w")
label=["GBB9032P","SLB9944P","SLP8772B","SKH5906G","SLZ300E","SLG9325A","SLU7747","SJY1928S","SLF4095B","SGK368Z","SJN7561J","SHA7822Z","SLG8804R","SMA2887K","SHC7520J","SLL8936S","SKC8287R","SKE5353S","SLM3669R","SJQ2807T","SLT3474Z","SLS7130U","SLN2539J","SKB9993J","SKQ9052X","SLQ6081U","SLJ8660Y","SLZ5663A","SLK7442L","SHB4493D","GBF5143M","SLM4222P","SLX8550H"]

for file1 in range(1,34):
	path="outnum/image"+str(file1)+".jpg"
	print(path)
	stri=number_detect(path,file1)
	
	if not stri==None and re.search(r"([A-Z]{2})+(\d{3})",stri):
		print(stri)
		F.write("image"+str(file1)+".jpg"+" = "+label[file1-1]+" = "+stri+" = "+"True"+"\n")
		
	else:
		F.write("image"+str(file1)+".jpg"+" = "+label[file1-1]+" = "+"None"+" = "+"True"+"\n")
