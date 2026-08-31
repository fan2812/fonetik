from django.apps import AppConfig

class PenelusuranFonetikConfig(AppConfig): # Sesuaikan nama class
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'penelusuran_fonetik'

    def ready(self):
        from django.core.cache import cache
        try:
            cache.clear()
        except Exception:
            pass