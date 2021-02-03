import cv2
import numpy as np


def getContours(image, canvas):
	contours, hierarchy = cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		area = cv2.contourArea(contour)
		if area > 1:
			cv2.drawContours(canvas, contour, -1, (255, 0, 0), 3)
			peri = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
			x, y, w, h = cv2.boundingRect(approx)
			cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 1)
	return canvas


if __name__ == '__main__':
	img = cv2.imread('resources/shapes3.png')
	img = cv2.resize(img, (450, 200))
	canvas = img.copy()

	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	blur = cv2.GaussianBlur(grey, ksize=(7, 7), sigmaX=1)
	canny = cv2.Canny(blur, 50, 50)
	canvas = getContours(canny, canvas)

	result = np.vstack([blur, canny])
	origin = np.vstack([img, canvas])

	cv2.imshow('Image', result)
	cv2.imshow('Shapes', origin)
	cv2.waitKey(0)
