from .base_request import BaseRequest
from model import body,endpoints
import requests

base_request = BaseRequest()

class FaceRegister():  
  def __init__(self):
    self.body = body
  def face_register_api(self,face_image,name,path):
    multipart_data = self.body.register_payload(face_image = face_image,user_name= name,path=path)
    r = requests.post(base_request.endpoints(api = endpoints.face_register),data=multipart_data,headers={'Content-Type': multipart_data.content_type})
    return r.content