import cv2
import numpy as np


if __name__ == '__main__':
	cv2.namedWindow('Tracked Bars')
	cv2.resizeWindow('Tracked Bars', 340, 80)

	cv2.createTrackbar('Hue Min', 'Tracked Bars', 0, 179, lambda v: v)
	cv2.createTrackbar('Hue Max', 'Tracked Bars', 19, 179, lambda v: v)

	while True:
		img = cv2.imread('resources/lena.jpg')
		imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		h_min = cv2.getTrackbarPos('Hue Min', 'Tracked Bars')
		h_max = cv2.getTrackbarPos('Hue Max', 'Tracked Bars')

		lower = np.array([h_min, h_min, h_min])
		upper = np.array([h_max, h_max, h_max])
		mask = cv2.inRange(imgHSV, lower, upper)

		mask = np.stack([mask, mask, mask], axis=-1)
		concat_img = np.hstack([img, imgHSV, mask])
		cv2.imshow('Result', concat_img)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
