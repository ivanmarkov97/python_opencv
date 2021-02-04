import cv2
import numpy as np


def preprocess(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=1)
	img = cv2.Canny(img, threshold1=150, threshold2=200)

	kernel = np.ones((5, 5), np.uint8)

	img = cv2.dilate(img, kernel=kernel, iterations=2)
	img = cv2.erode(img, kernel=kernel, iterations=1)
	return img


def draw_contours(img, canvas):
	biggest = None
	max_area = -np.float('inf')
	contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
	for contour in contours:
		area = cv2.contourArea(contour)
		if area > 3000:
			peri = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
			if area > max_area and len(approx) == 4:
				biggest = approx
				max_area = area
	cv2.drawContours(canvas, biggest, -1, color=(0, 255, 0), thickness=5)
	return biggest


def reorder_points(points):
	points = points.reshape(4, 2)
	new_points = np.zeros((4, 1, 2), np.int32)
	# pos with the smallest sum is the 0th position
	# pos with the biggest sum is the 3rd position
	# pos with min differ is the 1st position
	# pos with max differ is the 2nd position

	add_sum = points.sum(axis=1)
	new_points[0] = points[np.argmin(add_sum)]
	new_points[3] = points[np.argmax(add_sum)]

	diff_sum = np.diff(points, axis=1)
	new_points[1] = points[np.argmin(diff_sum)]
	new_points[2] = points[np.argmax(diff_sum)]

	# print(np.argmin(add_sum), np.argmin(diff_sum), np.argmax(add_sum), np.argmax(diff_sum))
	# ok 0 3 1 2

	return new_points


def get_warp(img, biggest, w, h):
	biggest = reorder_points(biggest)
	points1 = np.float32(biggest)
	points2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
	matrix = cv2.getPerspectiveTransform(points1, points2)
	output = cv2.warpPerspective(img, matrix, (w, h))
	return output


def main(config):
	TITLE = config['title']
	EXIT = config['exit']
	HEIGHT = config['height']
	WIDTH = config['width']

	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)

	while True:
		# True, cv2.imread('resources/document.jpg')
		success, frame = cap.read()
		# frame = cv2.resize(frame, (WIDTH, HEIGHT))
		frame = frame[30: frame.shape[0] - 30, :]  # crop boundaries
		canvas = frame.copy()
		wrap_output = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
		if success:
			frame = preprocess(frame)
			document_contour = draw_contours(frame, canvas)
			if document_contour is not None:
				wrap_output = get_warp(canvas, document_contour, frame.shape[1], frame.shape[0])
				# wrap_output = cv2.resize(wrap_output, (frame.shape[1], frame.shape[0]))
			frame = np.stack([frame, frame, frame], axis=-1)
			cv2.imshow(TITLE, np.hstack([frame, canvas, wrap_output]))
		if cv2.waitKey(1) & 0xFF == ord(EXIT) or not success:
			break
