import cv2
import numpy as np


colors = {
	'BLACK': {
		'h_min': 0,
		'h_max': 179,
		's_min': 0,
		's_max': 255,
		'v_min': 0,
		'v_max': 53
	}
}

all_points = []


def create_color_picker_hsv(name):
	cv2.namedWindow(name)
	cv2.createTrackbar('Hue MIN', name, 0, 179, lambda v: v)
	cv2.createTrackbar('Hue MAX', name, 0, 179, lambda v: v)
	cv2.createTrackbar('Sat MIN', name, 0, 255, lambda v: v)
	cv2.createTrackbar('Sat MAX', name, 0, 255, lambda v: v)
	cv2.createTrackbar('Value MIN', name, 0, 255, lambda v: v)
	cv2.createTrackbar('Value MAX', name, 0, 255, lambda v: v)


def draw_contour(img, canvas):
	contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
	for contour in contours:
		area = cv2.contourArea(contour)
		if area > 2000:
			cv2.drawContours(canvas, contour, -1, color=(0, 0, 255), thickness=1)
			peri = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
			x, y, w, h = cv2.boundingRect(approx)
			center_x = x + w // 2
			center_y = y + h // 2
			all_points.append((center_x, center_y))
			cv2.circle(canvas, (center_x, center_y), radius=10, color=(255, 255, 255), thickness=cv2.FILLED)
	return canvas


def draw_history(canvas):
	for point in all_points:
		cv2.circle(canvas, point, radius=10, color=(255, 255, 255), thickness=cv2.FILLED)


def main(config):
	TITLE = config['title']
	EXIT = config['exit']
	HEIGHT = config['height']
	WIDTH = config['width']
	USE_HSV = config['hsv']
	PAINT_COLOR = config['color']

	if PAINT_COLOR not in colors:
		print('Used the default color: black')
		color_params = colors['BLACK']
	else:
		color_params = colors[PAINT_COLOR]

	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)

	if USE_HSV:
		create_color_picker_hsv('HSV')

	while True:
		success, frame = cap.read()
		frame = frame[30: frame.shape[0] - 30, :]  # crop boundaries
		canvas = frame.copy()
		frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		if USE_HSV:
			h_min = cv2.getTrackbarPos('Hue MIN', 'HSV')
			h_max = cv2.getTrackbarPos('Hue MAX', 'HSV')
			s_min = cv2.getTrackbarPos('Sat MIN', 'HSV')
			s_max = cv2.getTrackbarPos('Sat MAX', 'HSV')
			v_min = cv2.getTrackbarPos('Value MIN', 'HSV')
			v_max = cv2.getTrackbarPos('Value MAX', 'HSV')

			lower = np.array([h_min, s_min, v_min])
			upper = np.array([h_max, s_max, v_max])
		else:
			lower = np.array([color_params['h_min'], color_params['s_min'], color_params['v_min']])
			upper = np.array([color_params['h_max'], color_params['s_max'], color_params['v_max']])
		mask = cv2.inRange(frame_HSV, lower, upper)
		canvas = draw_contour(mask, canvas)
		if success:
			if all_points:
				draw_history(canvas)
			cv2.imshow(TITLE, canvas)
		if cv2.waitKey(1) & 0xFF == ord(EXIT) or not success:
			break
