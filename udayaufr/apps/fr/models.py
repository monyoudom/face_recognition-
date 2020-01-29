from django.db import models
import os
from django.dispatch import receiver



# class Gender(models.Model):
#   gender  = models.CharField(max_length=7)
#   def __str__(self):
#       return str(self.gender)

class UserFaceDataset(models.Model):
  user_name = models.CharField(max_length=50)
  face_images = models.FileField(upload_to='fr/faces/')
  face_data = models.FileField(upload_to='fr/embeddings/')
  register_date = models.DateField(auto_now=True,auto_now_add=False)
  update        = models.DateField(auto_now=False,auto_now_add=True)
  def save(self, *args, **kwargs):
    if not (UserFaceDataset.objects.filter(user_name=self.user_name).exists()):
      super(UserFaceDataset, self).save(*args, **kwargs)
      return True
    else:
      
      return False


@receiver(models.signals.post_delete, sender=UserFaceDataset)
def auto_delete_file_on_delete(sender, instance, **kwargs):
  if instance.face_images and instance.face_data:
    if os.path.isfile(instance.face_images.path) and os.path.isfile(instance.face_data.path):
      os.remove(instance.face_images.path)
      os.remove(instance.face_data.path)


  


  
  
  





