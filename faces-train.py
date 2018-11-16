import cv2
import os
import numpy as np
import pickle
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

faceCascade = cv2.CascadeClassifier('/home/rohan/Documents/facialRecog/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

x_train = []
y_labels = []
current_id = 0
label_ids = {}

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpeg") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path).replace(" ", "_")).lower()
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id = label_ids[label]
			pil_image = Image.open(path).convert("L")
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")
			faces = faceCascade.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)
			for(x, y, w, h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id)
# print(label_ids)
with open("labels.pickle", "wb") as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")	