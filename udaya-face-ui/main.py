import cv2
import os
import sys
import numpy as np
from datetime import datetime
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi
from  api.face_register import FaceRegister
from api.search_face import  SearchFace
import json
from websocket import create_connection
from PIL import Image
import base64
import dlib
from imutils import face_utils
import time

ws = create_connection("ws://localhost:8001/websocket")
p = "/Users/thaungmonyodam/KIT/UDAYA/udaya-face-ui/face/encode/shape_predictor_68_face_landmarks.dat"

class USER(QDialog):        # Dialog box for entering name and key of new dataset.
    """USER Dialog """
    def __init__(self):
        super(USER, self).__init__()
        loadUi("UI/user_info.ui", self)

    def get_name(self):
        name = self.name_label.text()
       
        return name
    
class CAMERA(QDialog):        # Dialog box for entering name and key of new dataset.
    """USER Dialog """
    def __init__(self):
        super(CAMERA, self).__init__()
        loadUi("UI/camera.ui", self)

    def get_name_key(self):
        name = self.name_label.text()
        key = self.key_label.text()
        return name, key


class AUFR(QMainWindow):
    def __init__(self):
        super(AUFR, self).__init__()
        loadUi("UI/mainwindow.ui", self)
        # Classifiers, frontal face, eyes and smiles.
        self.face_classifier = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml") 
        self.eye_classifier = cv2.CascadeClassifier("classifiers/haarcascade_eye.xml")
        self.smile_classifier = cv2.CascadeClassifier("classifiers/haarcascade_smile.xml")
        
        # detect face using dlib 
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(p)
        
        # Variables
        self.camera_id = 0 # can also be a url of Video
        self.ret = False
        self.dataset_per_subject = 100
        self.name = "unknow"
        
        self.image = cv2.imread("icon/can.jpg", 1)
        self.modified_image = self.image.copy()
        self.draw_text("Face Recognition", 40, 30, 1, (255,255,255))
        self.display()
        self.image_encode = 0
        # Actions 
        self.register_btn.setCheckable(True)
        self.set_camera_btn.setCheckable(True)
        self.recognize_face_btn.setCheckable(True)
        self.search_face_btn.setCheckable(True)
        # Events
        self.recognize_face_btn.clicked.connect(self.recognize)
        self.register_btn.clicked.connect(self.generate)
        self.set_camera_btn.clicked.connect(self.camera_login)
        self.search_face_btn.clicked.connect(self.search_face)
        
    def start_timer(self):      # start the timeer for execution.
        self.capture = cv2.VideoCapture(self.camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.timer = QtCore.QTimer()
        if self.register_btn.isChecked():
            self.timer.timeout.connect(self.save_dataset)
        if self.recognize_face_btn.isChecked():
            self.timer.timeout.connect(self.update_image)
        if self.search_face_btn.isChecked():
            self.timer.timeout.connect(self.search)
        self.timer.start(5)
        
    def stop_timer(self):       # stop timer or come out of the loop.
        self.image = cv2.imread("icon/can.jpg", 1)
        self.modified_image = self.image.copy()
        self.draw_text("Face Recognition", 40, 30, 1, (255,255,255))
        self.display()
        self.timer.stop()
        self.ret = False
        self.capture.release()
        
    def convert_to_gbit(value):
        return value/1024./1024./1024.*8

    def send_stat(value):
        print ("%0.3f" % convert_to_gbit(value))
        
    def update_image(self):     # update canvas every time according to time set in the timer.
        if self.recognize_face_btn.isChecked():
            self.ret, self.image = self.capture.read()
            self.image = cv2.flip(self.image, 1)
            faces = self.get_faces()
            self.draw_rectangle(faces)
            start = time.time()
            for (x, y, w, h) in faces:
                name = self.recognize_web_socket(self.resize_image(self.image[y:y+h, x:x+w]))
                cv2.putText(self.image,name , (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 0), 1, cv2.LINE_AA)
        print("Neural network forward pass took {} seconds.".format(
                time.time() - start))
        self.display()
        
    def search_face(self):     # update canvas every time according to time set in the timer.
        if self.search_face_btn.isChecked():
            self.start_timer()
            self.search_face_btn.setText("Okay")
        else:
            search_face_call = SearchFace()
            data = search_face_call.search_face_api(face_image ="face.jpg" ,path="false")
            data = data.decode('utf8').replace("'", '"')
            data = json.loads(data)
            QMessageBox().about(self, "Face Register", data['data'][0]['msg'])
            self.search_face_btn.setText("Search")
            self.stop_timer()
 
    def search(self):
        if self.search_face_btn.isChecked():
            self.ret, self.image = self.capture.read()
            self.image = cv2.flip(self.image, 1)
            faces = self.get_faces()
            self.draw_rectangle(faces)
            if len(faces) is not 1 or len(faces) == 0:
                self.draw_text("Only One Person at a time or face not found")
            else:
                for (x, y, w, h) in faces:
                    cv2.imwrite('face.jpg', self.resize_image(self.image[y:y+h, x:x+w]))
    
        self.display()
        
    def display(self):      # Display in the canvas, video feed.
        try:
            pixImage = self.pix_image(self.image)
            self.video_feed.setPixmap(QtGui.QPixmap.fromImage(pixImage))
            self.video_feed.setScaledContents(True)
        except:
            print("dd")

    def pix_image(self, image): # Converting image from OpenCv to PyQT compatible image.
        qformat = QtGui.QImage.Format_RGB888  # only RGB Image
        if len(image.shape) >= 3:
            r, c, ch = image.shape
        else:
            r, c = image.shape
            qformat = QtGui.QImage.Format_Indexed8
        pixImage = QtGui.QImage(image, c, r, image.strides[0], qformat)
        return pixImage.rgbSwapped()
        
    def recognize(self):        # When recognized button is called.
        if self.recognize_face_btn.isChecked():
            self.start_timer()
            self.recognize_face_btn.setText("Stop")
        else:
            self.recognize_face_btn.setText("Recognize")
            self.stop_timer()
            
    def get_gray_image(self):       # Convert BGR image to GRAY image.
        try:
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            print("ddd")
    
    def get_faces(self):        # Get all faces in a image.
        # variables
        scale_factor = 1.1
        min_neighbors = 8
        min_size = (100, 100) 

        faces = self.face_classifier.detectMultiScale(
        					self.get_gray_image(),
        					scaleFactor = scale_factor,
        					minNeighbors = min_neighbors,
        					minSize = min_size)

        return faces
    def face_landmark_detection(self,image):
       
        # detect faces in the grayscale image
        rects = self.detector(image, 1)
        # loop over the face detections
        for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(image, rect)
            shape = face_utils.shape_to_np(shape)
        
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_image = image[y:y+h, x:x+w]
            if roi_image.size != 0:
                name = self.recognize_web_socket(self.resize_image(roi_image))
                cv2.putText(image, name, (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

            # print(encode_face)
            # data = cv2.imencode('.jpg', roi_face)[1].tostring()
            # socket_call(data)
        
            # show the face number
            
        
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            
            # for (x, y) in shape:
            #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            if self.register_btn.isChecked():
                cv2.imwrite('face.jpg', image)
        return image
    
    def save_dataset(self):     # Save images of new dataset generated using generate dataset button.

        if self.register_btn.isChecked():
            self.ret, self.image = self.capture.read()
            self.image = cv2.flip(self.image, 1)
            faces  = self.get_faces()
            self.draw_rectangle(faces)
            if len(faces) is not 1 or len(faces) == 0:
                self.draw_text("Only One Person at a time or face not found")
            else:
                for (x, y, w, h) in faces:
                    cv2.imwrite('face.jpg', self.resize_image(self.image[y:y+h, x:x+w]))
        else:
            self.register_btn.setChecked(False)
            face_register = FaceRegister()
            data = face_register.face_register_api(face_image ="face.jpg" ,name=self.name ,path="false")
            data = data.decode('utf8').replace("'", '"')
            data = json.loads(data)
            #print(data)
            QMessageBox().about(self, "Face Register", data['data'][0]['msg'])
            self.register_btn.setText("Register")
            self.stop_timer()
            
                  
        self.display()

    def draw_rectangle(self, faces):  
        for (x, y, w, h) in faces:
            # roi_gray_original = self.get_gray_image()[y:y + h, x:x + w]
            # roi_gray = self.resize_image(roi_gray_original, 92, 112)
            # roi_color = self.image[y:y+h, x:x+w]
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    def draw_text(self, text, x=20, y=20, font_size=2, color = (0, 255, 0)): # Draw text in current image in particular color.
        cv2.putText(self.image, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.6, color, font_size)
        
    def generate(self):     # Envoke user dialog and enter name and key.
        if self.register_btn.isChecked():
            try:
                user = USER()
                user.exec_()
                name = user.get_name()
                self.name = name
                self.start_timer()
                self.register_btn.setText("Generating")
            except:
                print("eroror",sys.exc_info()[0])
                msg = QMessageBox()
                msg.about(self, "User Information", '''Provide Information Please! \n name[string]\n key[integer]''')
                self.register_btn.setChecked(False)
                
    def camera_login(self):
        if self.set_camera_btn.isChecked():
            try:
                camera = CAMERA()
                camera.exec_()
                name, key = user.get_name_key()
                connect = 'rtsp://'+name+':'+key+'@192.168.1.64/1'
                self.set_camera_btn.isChecked(False)
            except:
                msg = QMessageBox()
                msg.about(self, "Error", '''Cannot connect''')
                self.set_camera_btn.isChecked(False)
  
                
    def recognize_web_socket(self,frame):
        # print(self.opencv_image_to_binary_image(self.image))
        _, img_encoded = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(img_encoded)
        result =  self.socket_call_api(jpg_as_text)
        return result
        
    
    def socket_call_api(self,frame):
        ws.send(frame)
        result =  ws.recv()
        return result
    
    def resize_image(self, image, width=180, height=180): # Resize image before storing.
        return cv2.resize(image, (width,height), interpolation = cv2.INTER_CUBIC)
                
    def about_info(self):       # Menu Information of info button of application.
        msg_box = QMessageBox()
        msg_box.setText('''
            AUFR (authenticate using face recognition) is an Python/OpenCv based
            face recognition application. It uses Machine Learning to train the
            model generated using haar classifier.
            Eigenfaces, Fisherfaces and LBPH algorithms are implemented.
            The code of this application is available at github @indian-coder.
        ''')
        msg_box.setInformativeText('''
            Ambedkar Institute of Technology, NCT of Delhi-110031.
            Mentor: Dr. Aatri Jain
            Team  : Md. Danish, Sumit Chaurasia
            September, 30th, 2018
            ''')
        msg_box.setWindowTitle("About AUFR")
        msg_box.exec_()
        
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = AUFR()         # Running application loop.
    ui.show()
    sys.exit(app.exec_())       #  Exit application.