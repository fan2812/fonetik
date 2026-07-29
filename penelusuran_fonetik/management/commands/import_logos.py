import py7zr
import tempfile
import os
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.contrib.auth.models import User
from penelusuran_fonetik.models import LogoDatabaseEntry, DatabaseChangeLog
from penelusuran_fonetik.ml_engine import extract_embedding, embedding_to_str

SUPPORTED = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tif', '.tiff'}


class Command(BaseCommand):
    help = 'Import logo dari file .7z ke database MySQL'

    def add_arguments(self, parser):
        parser.add_argument('archive', type=str, help='Path ke file .7z')
        parser.add_argument(
        '--admin',
        type=str,
        default='taufanstr',   # ← ganti ini
        help='Username admin yang melakukan import'
        )

    def handle(self, *args, **options):
        archive_path = options['archive']
        admin_name   = options['admin']
        filename     = archive_path.split('\\')[-1].split('/')[-1]

        if not os.path.exists(archive_path):
            self.stderr.write(self.style.ERROR(f'File tidak ditemukan: {archive_path}'))
            return

        # Ambil user admin untuk log
        try:
            admin_user = User.objects.get(username=admin_name)
        except User.DoesNotExist:
            admin_user = None
            self.stdout.write(f'User "{admin_name}" tidak ditemukan, log tanpa user.')

        rows_before = LogoDatabaseEntry.objects.count()
        sukses = gagal = skip = 0

        try:
            import tempfile
            tmpdir_obj = tempfile.TemporaryDirectory(dir='C:\\tmp')
            tmpdir = tmpdir_obj.name

            with py7zr.SevenZipFile(archive_path, mode='r') as z:
                    z.extractall(path=tmpdir)

            self.stdout.write('Ekstrak selesai. Mulai proses embedding...\n')

            for root, dirs, files in os.walk(tmpdir):
                    for fname in sorted(files):
                        ext = os.path.splitext(fname)[1].lower()
                        if ext not in SUPPORTED:
                            continue

                        filepath   = os.path.join(root, fname)
                        nama_merek = os.path.splitext(fname)[0].upper()

                        # Skip kalau sudah ada
                        if LogoDatabaseEntry.objects.filter(nama_merek=nama_merek).exists():
                            self.stdout.write(f'  SKIP : {nama_merek}')
                            skip += 1
                            continue

                        try:
                            with open(filepath, 'rb') as f:
                                img_bytes = f.read()

                            self.stdout.write(f'  Proses: {nama_merek}...')
                            emb = extract_embedding(img_bytes)

                            entry            = LogoDatabaseEntry()
                            entry.nama_merek = nama_merek
                            entry.embedding  = embedding_to_str(emb)
                            entry.logo_file.save(fname, ContentFile(img_bytes), save=True)

                            sukses += 1
                            self.stdout.write(self.style.SUCCESS(f'  OK    : {nama_merek}'))

                        except Exception as e:
                            gagal += 1
                            self.stderr.write(self.style.ERROR(f'  GAGAL : {fname} → {e}'))

            rows_after = LogoDatabaseEntry.objects.count()

            # Catat ke log
            DatabaseChangeLog.objects.create(
                admin_user  = admin_user,
                filename    = filename,
                fileformat  = '7z',
                rows_before = rows_before,
                rows_after  = rows_after,
                status      = 'success' if gagal == 0 else 'partial',
                notes       = f'Import logo via command. Sukses: {sukses}, Gagal: {gagal}, Skip: {skip}.'
            )

            self.stdout.write(self.style.SUCCESS(
                f'\nSelesai: {sukses} sukses | {gagal} gagal | {skip} dilewati | Total logo DB: {rows_after}'
            ))

        except Exception as e:
            rows_after = LogoDatabaseEntry.objects.count()

            DatabaseChangeLog.objects.create(
                admin_user  = admin_user,
                filename    = filename,
                fileformat  = '7z',
                rows_before = rows_before,
                rows_after  = rows_after,
                status      = 'failed',
                notes       = str(e)
            )

            self.stderr.write(self.style.ERROR(f'GAGAL TOTAL: {e}'))
