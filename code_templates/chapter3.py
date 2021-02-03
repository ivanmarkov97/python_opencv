import cv2
import numpy as np


if __name__ == '__main__':
	size = (320, 300, 3)
	img = np.zeros(size, np.uint8)

	cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), color=(255, 0, 0), thickness=3)
	cv2.rectangle(img, (30, 30), (img.shape[1] // 2, img.shape[0] // 2), color=(0, 255, 0), thickness=3)
	cv2.putText(img, 'text', (30, 25), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)

	cv2.imshow('Image', img)
	cv2.waitKey(0)
