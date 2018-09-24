# import necessary packages
import numpy as np
import cv2

def process_letter(thresh,output):	
	# assign the kernel size	
	kernel = np.ones((2,1), np.uint8) # vertical
	# use closing morph operation then erode to narrow the image	
	temp_img = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=3)
	# temp_img = cv2.erode(thresh,kernel,iterations=2)		
	letter_img = cv2.erode(temp_img,kernel,iterations=1)
	
	# find contours 
	(_,contours, _) = cv2.findContours(letter_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	# loop in all the contour areas
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(output,(x-1,y-5),(x+w,y+h),(0,255,0),1)

	return output	


for i in range(1,34):
	path1="gray_num2/num"+str(i)+".jpg"
	image1 = cv2.imread(path1)
	output1_letter = cv2.imread(path1)
	gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	ret1,th1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	output1_letter = process_letter(th1,output1_letter)
	path2="gray_num2_output/num"+str(i)+".jpg"
	cv2.imwrite(path2, output1_letter)	
