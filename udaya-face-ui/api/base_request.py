import requests
from model import endpoints,header


class BaseRequest():
  def __init__(self):
    self.base_url = header.base_url
    self.header  = header.content_type
  
  def request_header(self):
    return self.header
  
  def endpoints(self,api):
    url = self.base_url + api
    return url
  
    