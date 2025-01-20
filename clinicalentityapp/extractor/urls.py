from django.urls import path
from .views import clinical_extraction_view

urlpatterns = [
    path("", clinical_extraction_view, name="clinical_extraction_view"),  
]

