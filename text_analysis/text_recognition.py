# USAGE
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_01.jpg
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_04.jpg --padding 0.05

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2

def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < 0.1:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)




# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
net = cv2.dnn.readNet('frozen_east_text_detection.pb')



for i in range(1,34):
	if i!=7:
		path='num1/num'+str(i)+'.jpg'
		# load the input image and grab the image dimensions
		image = cv2.imread(path)
		orig = image.copy()
		(origH, origW) = image.shape[:2]

		# set the new width and height and then determine the ratio in change
		# for both the width and height
		(newW, newH) = (320, 320)
		rW = origW / float(newW)
		rH = origH / float(newH)

		# resize the image and grab the new image dimensions
		image = cv2.resize(image, (newW, newH))
		(H, W) = image.shape[:2]


		# construct a blob from the image and then perform a forward pass of
		# the model to obtain the two output layer sets
		blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
		net.setInput(blob)
		(scores, geometry) = net.forward(layerNames)

		# decode the predictions, then  apply non-maxima suppression to
		# suppress weak, overlapping bounding boxes
		(rects, confidences) = decode_predictions(scores, geometry)
		#(startX,startY,endX,endY)=
		boxes = non_max_suppression(np.array(rects), probs=confidences)
		if len(boxes)!=0:
			(startX,startY,endX,endY)=(boxes[0][0],boxes[0][1],boxes[0][2],boxes[0][3])
			cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
			path1='num1_output/num'+str(i)+'.jpg'
			cv2.imwrite(path1,image)
