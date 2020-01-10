import cv2
import numpy as np
import time
from websocket import create_connection
ws = create_connection("ws://192.168.11.36:8001/websocket")
face_cascade = cv2.CascadeClassifier(
    '../Document/Computer-Vision-with-Python/DATA/haarcascades/haarcascade_frontalface_default.xml')

def socket_call(frame):
    ws.send(str(frame))
    result =  ws.recv()
    return result

def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, 1.3, 5)
    print(face_rects)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        roi_face = face_img[y:y + h, x:x + w]
        cv2.imwrite("image.jpg",roi_face)
        data = cv2.imencode('.jpg', roi_face)[1].tostring()
        result =  socket_call(data)
        print(result)
    return face_img

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
while True:
    ret, frame = cap.read(0)
    frame = detect_face(frame)
    cv2.imshow('Video Capture', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()