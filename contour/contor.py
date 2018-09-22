
import numpy as np
import cv2

for file1 in range(1,34):
	b2e='blur2edge/num'+str(file1)+'.jpg'
	can='canny/num'+str(file1)+'.jpg'
	normal_path='num2/num'+str(file1)+'.jpg'
	
	img1=cv2.imread(b2e)
	img2=cv2.imread(can)
	img3=cv2.imread(normal_path)
	
	imgray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	imgray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	imgray3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
	
	ret1,thresh1 = cv2.threshold(imgray1,127,255,0)
	ret2,thresh2 = cv2.threshold(imgray2,127,255,0)
	ret3,thresh3 = cv2.threshold(imgray3,127,255,0)
	
	im1, contours1, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	im2, contours2, hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	im3, contours3, hierarchy3 = cv2.findContours(thresh3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	
	cv2.drawContours(img1, contours1, -1, (0,255,0), 3)
	cv2.drawContours(img2, contours2, -1, (0,255,0), 3)
	cv2.drawContours(img3, contours3, -1, (0,255,0), 3)
	
	b2e2c='blur2edge2contour/num'+str(file1)+'.jpg'
	c2c='canny2contour/num'+str(file1)+'.jpg'
	nc='normal_contour/num'+str(file1)+'.jpg'

	cv2.imwrite(b2e2c,img1)
	cv2.imwrite(c2c,img2)
	cv2.imwrite(nc,img3)




