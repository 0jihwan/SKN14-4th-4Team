from django.urls import path
from . import views

urlpatterns = [
    path('', views.main, name='main'),
    path('chat/', views.chat, name='chat'),
    path('signup/', views.signup, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('clear_history/', views.clear_history, name='clear_history'),
]
