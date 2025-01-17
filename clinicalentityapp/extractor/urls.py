from django.urls import path
from . import views

urlpatterns = [
    path("", views.clinical_extraction_view, name="clinical_extraction"),
]


