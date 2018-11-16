import cv2
import os
import sys

name = sys.argv[1]
count = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
image_dir = os.path.join(image_dir, name)

if not os.path.exists(image_dir):
	os.makedirs(image_dir)

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	cv2.imwrite(os.path.join(image_dir, "%d.jpg" % count), frame)
	count += 1
	cv2.imshow('frame', frame)
	if(cv2.waitKey(20) & 0xFF == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()