from requests_toolbelt.multipart.encoder import MultipartEncoder


def register_payload(face_image,user_name,path):
  multipart_data = MultipartEncoder(
    fields={
    'face_image' : ('face.jpg', open(face_image, 'rb')),
    'user_name'  : user_name,
    'path'       : path
      }
  )
  return multipart_data

def search_face_payload(face_image,path):
  multipart_data = MultipartEncoder(
    fields={
      'face_image' : ('face.jpg', open(face_image, 'rb')),
      'path'       : path
    }
  )
  return multipart_data