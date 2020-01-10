import cv2
from PIL import Image
from .utils import (save_embedding,identify_face,face_dlib ,embedding_caluation,covert_to_tensor)
from torchvision import transforms
import numpy as np
from scipy.misc import imread
import base64



def get_embedding(face):
  embedding = embedding_caluation(face)
  return embedding.detach().numpy()
  
def insert_embedding(face,name,path):
  embedding = get_embedding(face)
  validation = search_face(face)
  if validation == "unknow":
    save_embedding(embedding=embedding, filename=name, embeddings_path=path)
    return name+".npy"
  else:
    return False
   
def search_face(face):
  embedding = get_embedding(face)
  # Compare euclidean distance between this embedding and the embeddings in 'embeddings/'
  name = identify_face(embedding=embedding) 
  return name

def face_live_p_1(frame):
  jpg_original = base64.b64decode(frame)
  nparr = np.fromstring(jpg_original, np.uint8)
  img = cv2.imdecode(nparr, cv2.COLOR_RGB2BGR) 
  face = face_dlib(img)
  if face is None:
    return "searching"
  face  = covert_to_tensor(face)
  name = search_face(face)
  return name


  



      



