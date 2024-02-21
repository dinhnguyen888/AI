import cv2
face_cascade = cv2.CascadeClassifier('cars.xml')
vc = cv2.VideoCapture('road.avi')