import cv2
import glob
for i in glob.glob("num1/*.jpg"):
	name=i[5:]
	img=cv2.imread(i)
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	path1="gray_num1/"+name
	cv2.imwrite(path1,img)

	
