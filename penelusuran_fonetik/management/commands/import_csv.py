import pandas as pd
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from penelusuran_fonetik.models import DataMerek, DatabaseChangeLog

class Command(BaseCommand):
    help = 'Import merek dari CSV ke database MySQL'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path ke file CSV')
        parser.add_argument('--admin', type=str, default='taufanstr')

    def handle(self, *args, **options):
        csv_path   = options['csv_file']
        admin_name = options['admin']
        filename   = csv_path.split('\\')[-1].split('/')[-1]

        try:
            admin_user = User.objects.get(username=admin_name)
        except User.DoesNotExist:
            admin_user = None

        rows_before = DataMerek.objects.count()

        try:
            df = pd.read_csv(csv_path, sep=None, engine='python')
            
            target_col = df.columns[0]
            for col in df.columns:
                if 'merek' in str(col).lower() or 'brand' in str(col).lower():
                    target_col = col
                    break

            # 1. Kumpulkan semua nama dari CSV
            values = {
                v.strip().upper()
                for v in df[target_col].dropna().astype(str)
                if v.strip()
            }

            # 2. Cek mana yang benar-benar baru (belum ada di DB)
            existing = set(DataMerek.objects.values_list('nama_merek', flat=True))
            new_names = values - existing

            # 3. Siapkan datanya
            new_objects = [DataMerek(nama_merek=nama) for nama in new_names]
            sukses = len(new_objects)

            # 4. MASUKKAN SEKALIGUS (Ini TIDAK akan memicu log berkali-kali)
            if new_objects:
                DataMerek.objects.bulk_create(new_objects, batch_size=1000)

            rows_after = DataMerek.objects.count()

            # 5. BUAT 1 BARIS LOG SAJA DI SINI
            DatabaseChangeLog.objects.create(
                admin_user  = admin_user,
                filename    = filename,
                fileformat  = 'csv',
                rows_before = rows_before,
                rows_after  = rows_after,
                status      = 'success',
                # Pesan ini akan persis seperti yang Anda mau:
                notes       = f'Upload berhasil via Terminal. Ditambah {sukses} data baru.'
            )

            self.stdout.write(self.style.SUCCESS(f'\nSelesai! 1 Baris log dibuat untuk {sukses} data baru.'))

        except Exception as e:
            self.stderr.write(self.style.ERROR(f'GAGAL: {e}'))