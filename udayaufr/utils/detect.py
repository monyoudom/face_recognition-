import cv2
from .utils import (save_embedding,identify_face,face_dlib ,embedding_caluation,covert_to_tensor)
import numpy as np
from scipy.misc import imread
import base64


def get_embedding(face):
  embedding = embedding_caluation(face)
  return embedding.detach().numpy()
  
def insert_embedding(face,name,path):
  embedding = get_embedding(face)
  save_embedding(embedding=embedding, filename=name, embeddings_path=path)
  return name+".npy"
   
def search_face(face):
  embedding = get_embedding(face)
  # Compare euclidean distance between this embedding and the embeddings in 'embeddings/'
  name = identify_face(embedding=embedding) 
  return name

def face_live_p_1(frame):
  frame = frame.replace('data:image/jpeg;base64,','')
  jpg_original = base64.b64decode(frame)
  face_image = 'face.jpg'
  with open(face_image, 'wb') as f:
    f.write(jpg_original)
  face = imread(name=face_image, mode='RGB')
  face = face_dlib(face)
  if face is None:
    return "1"
  face  = covert_to_tensor(face)
  name = search_face(face)
  if name == "unknow":
    return "0"
  else:
    return "2-"+name


  



      



