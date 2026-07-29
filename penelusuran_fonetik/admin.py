from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from django.shortcuts import redirect
from django.contrib.auth import logout 

# --- 1. KEAMANAN: STAFF & SUPERUSER BISA AKSES ---
# Sekarang Staff biasa tidak akan melihat pesan "Not Authorized" 
# selama kotak 'Staff Status' mereka sudah dicentang.
admin.site.has_permission = lambda request: request.user.is_active and request.user.is_staff

# --- 2. CUSTOM USER ADMIN: REDIRECT & LOGOUT SETELAH SAVE ---
try:
    admin.site.unregister(User)
except admin.sites.NotRegistered:
    pass

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    def response_change(self, request, obj):
        """
        Setelah Superuser selesai mengedit user (misal memberi akses Staff) 
        dan menekan Save, sesi akan dibersihkan agar tidak 'nyangkut'.
        """
        res = super().response_change(request, obj)
        
        if "_continue" not in request.POST and "_addanother" not in request.POST:
            # Tetap lakukan logout paksa agar transisi antar akun tetap bersih
            logout(request) 
            return redirect('admin:login')
        
        return res