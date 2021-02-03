import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

dilate_kernel = np.ones((3, 3), np.uint8)

while True:
	success, frame = cap.read()
	if success is True:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blured = cv2.GaussianBlur(gray, (7, 7), 0)
		canny = cv2.Canny(frame, threshold1=50, threshold2=50) # edge detection
		dilate = cv2.dilate(canny, dilate_kernel, iterations=2)
		cv2.imshow('Video', dilate)
	if cv2.waitKey(1) & 0xFF == ord('q') or success is False:
		break
