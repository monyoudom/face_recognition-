from django.urls import path
from .views import register ,search


app_name = 'fr'
urlpatterns = [
    path('face/register/',register , name='register'),
    path('face/search/', search, name='search'),
]