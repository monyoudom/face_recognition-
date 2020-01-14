
#!/usr/bin/env python3
from facenet_pytorch import MTCNN, InceptionResnetV1, prewhiten, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import tensorflow as tf
import numpy as np
import glob
import os
from scipy.misc import imresize, imsave
from collections import defaultdict
# from sklearn.svm import SVC
import cv2
import imutils
import math
import dlib
from imutils import face_utils
from torchvision.transforms import functional as F
import time

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
root_path = APP_ROOT.replace('/utils','')
# resnet = InceptionResnetV1(classify=True, num_classes=1001).eval()

shape_predictor_68_face_landmarks  = APP_ROOT + "/model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor()

# path of embedding
path_embedding = root_path  + "/media/fr/embeddings/"


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def face_dlib(image):
    rects = detector(image, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
        # (x, y, w, h) = face_utils.rect_to_bb(rect)
        # roi_image = image[y:y+h, x:x+w]
        face = extract_face(image,face_utils.rect_to_bb(rect))
        return face 

def extract_face(img, box, image_size=160, margin=1):
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]

    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img.size)),
        int(min(box[3] + margin[1] / 2, img.size)),
    ]

    
    face = img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
    face = cv2.resize(face, (image_size,image_size), interpolation = cv2.INTER_CUBIC)
    return face
    
def covert_to_tensor(image):
    face = F.to_tensor(np.float32(image))
    img = fixed_image_standardization(face)
    return img.unsqueeze(0)

def embedding_caluation(face_image):
    embedding = resnet(face_image)
    return embedding


def distance_face(emb1, emb2):
    return np.sqrt(((emb1 - emb2) ** 2).sum())

    
def save_embedding(embedding, filename, embeddings_path):
    
    # Save embedding of image using filename
    path = os.path.join(embeddings_path, str(filename))
    try:
        np.save(path, embedding)

    except Exception as e:
        print(str(e))
        
def remove_file_extension(filename):
   
    filename = os.path.splitext(filename)[0]
    return filename

def identify_face(embedding):
    # start = time.time()
    min_distance = 100
    result = "unknow"
    embeddings = []
    names = []
    try:
        for embedding_file in glob.iglob(path_embedding+"*.npy"):
            database_embedding = np.load(embedding_file)
            distance = distance_face(embedding,database_embedding)
            if distance < min_distance:
                min_distance = distance
                result =  remove_file_extension(embedding_file)
                result = result.replace(path_embedding,'')  
        proba = face_distance_to_conf(min_distance)
        # print("Neural network forward pass took {} seconds.".format(
        #         time.time() - start))
        print(proba,embeddings)
        if proba > 0.90:
            return result+" {0:.2f}".format(proba*100)+"%"
        return "unknow"
    except Exception as e:
        print(e)
        return "unknow"


def face_distance_to_conf(face_distance, face_match_threshold=0.85):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))