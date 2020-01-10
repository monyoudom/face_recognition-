from django.contrib import admin
from .models import UserFaceDataset

# Register your models here.
class UserFaceAdmin(admin.ModelAdmin):
    list_display = ('user_name','face_data',)
admin.site.register(UserFaceDataset,UserFaceAdmin)


# class GenderAdmin(admin.ModelAdmin):
#     list_display = ('gender',)
# admin.site.register(Gender,GenderAdmin)
