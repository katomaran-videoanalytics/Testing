import cv2

for i in range(1,34):
	path="denoised_image/num"+str(i)+".jpg"
	print(path)
	img=cv2.imread(path)
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	path1="denoising2gray/num"+str(i)+".jpg"
	cv2.imwrite(path1,img)
