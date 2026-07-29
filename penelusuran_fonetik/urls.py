from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static
from penelusuran_fonetik import views

urlpatterns = [
    path('admin/',         admin.site.urls),
    path('',               views.search_view,       name='search_view'),
    path('login/',         auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('register/',      views.register,           name='register'),
    path('logout/',        views.logout_view,        name='logout'),
    path('upload/',        views.upload_database,    name='upload_database'),
    path('log/',           views.database_log_view,  name='database_log'),
    path('session-check/', views.session_check_view, name='session_check'),
    path('staff-check/',   views.staff_check_view,   name='staff_check'),
    path('logo/',                      views.logo_search_view, name='logo_search'),
    path('logo/upload/',               views.logo_upload_view, name='logo_upload'),
    path('logo/delete/<int:logo_id>/', views.logo_delete_view, name='logo_delete'),
    path('api/v1/', include('penelusuran_fonetik.api_urls', namespace='api')),
    path('merek/edit/<int:merek_id>/', views.merek_edit_view, name='merek_edit'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)