from .base_request import BaseRequest
from model import body,endpoints
import requests

base_request = BaseRequest()

class SearchFace():  
  def __init__(self):
    self.body = body
  def search_face_api(self,face_image,path):
    multipart_data = self.body.search_face_payload(face_image = face_image,path=path)
    r = requests.post(base_request.endpoints(api = endpoints.search_face),data=multipart_data,headers={'Content-Type': multipart_data.content_type})
    return r.content