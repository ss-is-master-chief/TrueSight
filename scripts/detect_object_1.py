# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np

import imutils
import time
import cv2
import datetime

def detect_object(prototxt, model, confidence, image):
		# construct the argument parse and parse the arguments

		# initialize the list of class labels MobileNet SSD was trained to
		# detect, then generate a set of bounding box colors for each class
		CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]
		COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

		# load our serialized model from disk
		print("[INFO] loading model...")
		net = cv2.dnn.readNetFromCaffe(prototxt, model)

		# initialize the video stream, allow the cammera sensor to warmup,
		# and initialize the FPS counter
		print("[INFO] starting video stream...")
		# vs = VideoStream(src=0).start()
		# time.sleep(5.0)
		# fps = FPS().start()

		# loop over the frames from the video stream
		#while True:
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 400 pixels
		#frame = vs.read()
		frame = cv2.imread(image)
		frame = imutils.resize(frame, width=600)

		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			conf = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if conf > confidence:
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the prediction on the frame
				label = "{}: {:.2f}%".format(CLASSES[idx],
					conf * 100)
				if(CLASSES[idx]=='person'):
					print(CLASSES[idx], conf * 100, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), (endX-startX)*(endY*startY))

				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey()

		# if the `q` key was pressed, break from the loop
		# if key == ord("q"):
		# 	break

		# update the FPS counter
		#fps.update()

	# stop the timer and display FPS information
	#fps.stop()


	# do a bit of cleanup

cv2.destroyAllWindows()
	#vs.stop()