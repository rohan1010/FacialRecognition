import numpy as np
import cv2
import pickle

faceCascade = cv2.CascadeClassifier('/home/rohan/Documents/facialRecog/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
with open("labels.pickle", "rb") as f:
	old_labels = pickle.load(f)
	labels = {v:k for k, v in old_labels.items()}

color = (255, 0, 0)
stroke = 2
font = cv2.FONT_HERSHEY_SIMPLEX
text_color = (255, 255, 255)

cap = cv2.VideoCapture(0)

while (True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
	for(x, y, w, h) in faces:
		roi_gray = gray[y:y+h, x:x+w]
		id, conf = recognizer.predict(roi_gray)
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
		# print(id)
		# print(labels[id])
		name = labels[id]
		cv2.putText(frame, name, (x, y), font, 1, text_color, stroke, cv2.LINE_AA)

	cv2.imshow('frame', frame)
	if(cv2.waitKey(20) & 0xFF == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()