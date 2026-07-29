from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from . import api_views

app_name = 'api'

urlpatterns = [
    # --- Auth (login mobile, dapat access + refresh token) ---
    path('auth/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # --- Search ---
    path('search/fonetik/', api_views.api_search_fonetik, name='search_fonetik'),
    path('search/logo/', api_views.api_search_logo, name='search_logo'),
]