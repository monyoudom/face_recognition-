
from .models import UserFaceDataset
from django.http import JsonResponse
import numpy as np
import cv2
import urllib.request
from .models import UserFaceDataset
from os.path import join
from utils import detect
from django.conf import settings
from  utils.utils import face_dlib,covert_to_tensor
from utils import statuscode
from django.views.decorators.csrf import csrf_exempt
from django.db import IntegrityError
import json
from django.http import HttpResponse
from scipy.misc import imread

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

@csrf_exempt
def register(request):
      user_face = UserFaceDataset()
      if request.method == "POST":
        user_name  = request.POST['user_name']
        path       = request.POST['path']
  
        # varible for upload image
        upload_face_image = join(settings.MEDIA_ROOT,'fr/faces/'+str(user_name)+'.png')
        face_embedding = join(settings.MEDIA_ROOT,'fr/embeddings/')
          
        # varible to insert path to db
        face_path = "fr/faces/"+str(user_name)+'.png'
        embedding_path_store = "fr/embeddings/"
       
        try:
          if path == "true":
            #load the image from url path  
            face_image = request.POST['face_image']   
            req = urllib.request.urlopen(face_image)
            image = np.asarray(bytearray(req.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.COLOR_RGB2BGR)
            if image is None:
              return  JsonResponse(status=400, data={"status": 500,"info": "failed","data": [{"msg": "have no eye"}]})
          else:
            face_image = request.FILES['face_image'] 
            image = np.asarray(bytearray(face_image.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.COLOR_RGB2BGR)

          face = face_dlib(image)
          if face is None:
            return JsonResponse({'errors': {'error_message': 'No face detected'}})

          face_tensor = covert_to_tensor(face)        
          embedding_path = detect.insert_embedding(face_tensor, user_name,face_embedding)
          if embedding_path == False:
            return JsonResponse({'errors': {'error_message': 'cannot recognize'}})
          cv2.imwrite(upload_face_image,face)
          # insert to db    
          user_face.face_data.name =  embedding_path_store+embedding_path 
          user_face.face_images.name = face_path  
          user_face.user_name = user_name
          if user_face.save():
            return JsonResponse(status=200, data={"status": 200,"info": "success","data": [{"msg": "user face add to database"}]})
          else: 
            return JsonResponse(status=200, data={"status": 400,"info": "failed","data": [{"msg": "user already exited"}]})
   
        except Exception as e:
          return JsonResponse(status=200, data={"status": 500,"info": "fail","data": [{"msg": str(e)}]})
      else:
        return JsonResponse(status=200, data={"status": 500,"info": "fail","data": [{"msg": "allow only post"}]})
            

@csrf_exempt   
def search(request):
    
  if request.method == "POST":
    path = request.POST['path']
    try:
      if path == "true":
        #load the image from url path  
        face_image = request.POST['face_image']
        req = urllib.request.urlopen(face_image)
        image = np.asarray(bytearray(req.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
      else:
        face_image = request.FILES['face_image']
        image = imread(name=face_image, mode='RGB')
        
      # face detection    
      face = face_dlib(image)
      if face is None:
        return JsonResponse({'errors': {'error_message': 'No face detected or Eyes detected'}})
      
      face_tensor = covert_to_tensor(face)
      replace = join(settings.MEDIA_ROOT,'fr/embeddings/')
      # recognition
      user_name = detect.search_face(face_tensor)
      user_name = user_name.replace(replace,'')
              
      return JsonResponse({"status": 200,"info": "success","data": [{"msg": user_name}]})
    except Exception as e:
      return JsonResponse({'errors': {'error_message': str(e)}})   
  else:        
    return JsonResponse(status=400, data={"status": "500","info": "fail","data": [{"msg": "allow only post"}]})