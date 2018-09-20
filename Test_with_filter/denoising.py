import cv2 as cv
for i in range(1,34):
	path="gray_image/num"+str(i)+".jpg"
	print(path)
	img = cv.imread(path)
	b,g,r = cv.split(img)           # get b,g,r
	rgb_img = cv.merge([r,g,b])     # switch it to rgb
	# Denoising
	dst = cv.fastNlMeansDenoising(img,None,10,7,21) #For RGB-
	#dst=cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
	b,g,r = cv.split(dst)           # get b,g,r
	rgb_dst = cv.merge([r,g,b])     # switch it to rgb
	path1="grayimage2denoising/num"+str(i)+".jpg"
	cv.imwrite(path1,rgb_dst)
