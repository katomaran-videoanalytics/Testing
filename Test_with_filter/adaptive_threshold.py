
import cv2
import numpy as np
for i in range(1,34):
	path="grayimage2denoising/num"+str(i)+".jpg"
	print(path)
	img=cv2.imread(path,0)
	th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
	path1="grayimage2denoising2threshold/num"+str(i)+".jpg"
	cv2.imwrite(path1,th3)

