from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

# Flag global untuk nonaktifkan signal saat upload massal
# Diset True oleh views.py/management command sebelum loop, lalu False lagi sesudahnya
_disable_signals = False


# --- MODEL 1: LOG PERUBAHAN ---
class DatabaseChangeLog(models.Model):
    admin_user  = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    timestamp   = models.DateTimeField(auto_now_add=True)
    filename    = models.CharField(max_length=255)
    fileformat  = models.CharField(max_length=10, default='unknown')
    rows_before = models.IntegerField(default=0)
    rows_after  = models.IntegerField(default=0)
    status      = models.CharField(max_length=20)  # success, failed, warning
    notes       = models.TextField(blank=True, default='')

    class Meta:
        ordering        = ['-timestamp']
        verbose_name    = "Database Change Log"
        verbose_name_plural = "Database Change Logs"

    def __str__(self):
        user_display = self.admin_user.username if self.admin_user else "System"
        return f"{user_display} - {self.filename} - {self.timestamp.strftime('%d/%m/%Y %H:%M')}"

    @property
    def row_change(self):
        return self.rows_after - self.rows_before


# --- MODEL 2: DATA MEREK ---
class DataMerek(models.Model):
    nama_merek = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name        = "Data Merek"
        verbose_name_plural = "Data Merek"

    def __str__(self):
        return self.nama_merek


# --- MODEL 3: LOGO DATABASE ---
class LogoDatabaseEntry(models.Model):
    nama_merek = models.CharField(max_length=255, verbose_name="Nama Merek")
    logo_file  = models.ImageField(upload_to='logos/', verbose_name="File Logo")
    embedding  = models.TextField(verbose_name="Embedding (JSON)", blank=True, default='[]')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering            = ['nama_merek']
        verbose_name        = "Logo Terdaftar"
        verbose_name_plural = "Logo Terdaftar"

    def __str__(self):
        return self.nama_merek

    def get_logo_url(self):
        return self.logo_file.url if self.logo_file else None


# --- SIGNALS ---
# Hanya aktif untuk operasi manual (Django Admin, shell)
# Dinonaktifkan saat upload massal dari web/command supaya tidak dobel

@receiver(post_save, sender=DataMerek)
def log_merek_save(sender, instance, created, **kwargs):
    pass


@receiver(post_delete, sender=DataMerek)
def log_merek_delete(sender, instance, **kwargs):
    # Skip kalau flag aktif
    if _disable_signals:
        return

    current_count = DataMerek.objects.count()
    DatabaseChangeLog.objects.create(
        admin_user  = None,
        filename    = "Django Admin",
        fileformat  = "MySQL",
        rows_before = current_count + 1,
        rows_after  = current_count,
        status      = "warning",
        notes       = f"Otomatis: Hapus merek '{instance.nama_merek}'"
    )