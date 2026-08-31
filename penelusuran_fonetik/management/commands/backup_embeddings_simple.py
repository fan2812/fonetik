"""
FILE BARU: penelusuran_fonetik/management/commands/backup_embeddings_simple.py

Backup embedding SEMUA logo ke file JSON, sebelum regenerate_embeddings dijalankan.
Simpel: cuma dump id, nama_merek, embedding ke satu file dengan timestamp.

Cara pakai:
    python manage.py backup_embeddings_simple

Untuk rollback manual nanti (kalau perlu), file JSON ini bisa dibaca lagi
dan di-load balik ke kolom embedding tiap LogoDatabaseEntry.
"""
import json
import os
from datetime import datetime
from django.core.management.base import BaseCommand
from django.conf import settings
from penelusuran_fonetik.models import LogoDatabaseEntry


class Command(BaseCommand):
    help = 'Backup embedding semua logo ke file JSON sebelum regenerate'

    def handle(self, *args, **options):
        entries = LogoDatabaseEntry.objects.all()
        total = entries.count()

        backup_data = []
        for entry in entries:
            backup_data.append({
                'id': entry.id,
                'nama_merek': entry.nama_merek,
                'embedding': entry.embedding,
            })

        backup_dir = os.path.join(settings.BASE_DIR, 'backups')
        os.makedirs(backup_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(backup_dir, f'embeddings_backup_{timestamp}.json')

        with open(filepath, 'w') as f:
            json.dump(backup_data, f)

        self.stdout.write(self.style.SUCCESS(
            f'Backup selesai: {total} embedding disimpan ke {filepath}'
        ))