from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('report/pdf/', views.download_report_pdf, name='report_pdf'),
]
