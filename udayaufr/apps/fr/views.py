
from .models import UserFaceDataset
from django.http import JsonResponse
import numpy as np
from .models import UserFaceDataset
from os.path import join
from django.conf import settings
from  utils.utils import face_dlib,covert_to_tensor
from utils import statuscode,detect
from django.views.decorators.csrf import csrf_exempt
from django.db import IntegrityError
from django.http import HttpResponse
from scipy.misc import imread
from django.core.files.storage import default_storage


@csrf_exempt
def register(request):

      user_face = UserFaceDataset()
      if request.method == "POST":
        user_name  = request.POST['user_name']

        # varible for upload image
        face_embedding = join(settings.MEDIA_ROOT,'fr/embeddings/')
          
        # varible to insert path to db
        face_path = "fr/faces/"+str(user_name)+'.png'
        embedding_path_store = "fr/embeddings/"
       
        try:
          face_image = request.FILES['face_image']
          image = imread(name=face_image, mode='RGB')
          face = face_dlib(image)
          if face is None or len(face) > 2 :
            return JsonResponse(status=200, data={"status": 500,"info": "failed","data": [{"msg": "No face detected or face more than two"}]})

          face_tensor = covert_to_tensor(face)     
          embedding_path = detect.insert_embedding(face_tensor, user_name,face_embedding)
          if embedding_path == False:
            return JsonResponse(status=200, data={"status": 500,"info": "failed","data": [{"msg": "This face already register"}]})
          
          path = default_storage.save(face_path, request.FILES['face_image'])
          
          # insert to db    
          user_face.face_data =  embedding_path_store+embedding_path 
          user_face.face_images = path
          user_face.user_name = user_name
          if user_face.save() == False:
           UserFaceDataset.objects.filter(user_name=user_name).update(face_images=path,face_data= embedding_path_store+embedding_path) 
          
          return JsonResponse(status=200, data={"status": 200,"info": "success","data": [{"msg": "user face add to database"}]})
        except Exception as e:
          return JsonResponse(status=200, data={"status": 500,"info": "fail","data": [{"msg": str(e)}]})
      else:
        return JsonResponse(status=200, data={"status": 500,"info": "fail","data": [{"msg": "allow only post"}]})


    
@csrf_exempt   
def search(request):
    
  if request.method == "POST":
    try:
      face_image = request.FILES['face_image']
      image = imread(name=face_image, mode='RGB')
        
      # face detection    
      face = face_dlib(image)
      if face is None:
        return JsonResponse(status=200, data={"status": 500,"info": "failed","data": [{"msg": "No face detected or Eyes detected"}]})
      
      face_tensor = covert_to_tensor(face)
      replace = join(settings.MEDIA_ROOT,'fr/embeddings/')
      # recognition
      user_name = detect.search_face(face_tensor)
      user_name = user_name.replace(replace,'')
              
      return JsonResponse({"status": 200,"info": "success","data": [{"msg": user_name}]})
    except Exception as e:
      return JsonResponse(status=200, data={"status": 500,"info": "failed","data": [{"msg": str(e)}]})  
  else:        
    return JsonResponse(status=200, data={"status": "500","info": "fail","data": [{"msg": "allow only post"}]})