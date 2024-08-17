import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    if img is None:
        print("Error:empty image")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return img
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img 

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error:could not open the camera")
    exit()
    

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error:failed to capture image from camera.")
        break
    
    frame = detect_face(frame)
    if frame is None or frame.size == 0:
        print("Error: Empty frame returned from detect_face.")
        continue

    cv2.imshow('Video Face Detection', frame)

    if cv2.waitKey(20) & 0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()



