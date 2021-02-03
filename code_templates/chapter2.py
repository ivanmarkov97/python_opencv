import cv2


if __name__ == '__main__':
	img = cv2.imread('resources/lena.jpg')
	size = (320, 300)
	img = cv2.resize(img, size)
	img = img[13:287, 13:287]
	cv2.imshow('Lena', img)
	cv2.waitKey(0)
