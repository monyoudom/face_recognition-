from imutils import face_utils
import numpy as np
import argparse
import imutils
from facenet_pytorch import MTCNN, InceptionResnetV1, prewhiten, training
import torch
import cv2
import dlib
from torchvision.transforms import functional as F
# from websocket import create_connection
# ws = create_connection("ws://192.168.11.36:8001/websocket")

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument('-p', '--shape-predictor', required=True,
# 	help='path to facial landmark predictor')
# ap.add_argument('-i', '--image', required=True,
# 	help='path to input image')
# args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "/Users/thaungmonyodam/KIT/UDAYA/udaya-face-ui/face/encode/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# predictor = dlib.shape_predictor(args["shape_predictor"])

# def socket_call(frame):
#     ws.send(str(frame))
#     result =  ws.recv()
#     return result



def face_landmark_detection(image):
    image = imutils.resize(image, width=600)
    image = imutils.resize(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_image = image[y:y+h, x:x+w]
        # img = torch.from_numpy(roi_image).float().to(device)
        face = F.to_tensor(np.float32(roi_image))
        img = fixed_image_standardization(face)
        embedding = resnet(img.unsqueeze(0))
        print(embedding)
        

        # print(encode_face)
        # data = cv2.imencode('.jpg', roi_face)[1].tostring()
        # socket_call(data)
    
        # show the face number
        cv2.putText(image, 'Face #{}'.format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    return image

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
while True:
    ret, frame = cap.read()
    frame = face_landmark_detection(frame)
    cv2.imshow('Video Capture', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()