"""
FILE BARU: penelusuran_fonetik/management/commands/regenerate_embeddings.py

Regenerate embedding SEMUA logo di LogoDatabaseEntry menggunakan
pipeline ml_engine.py yang sekarang (preprocessing grayscale).

WAJIB dijalankan setelah ml_engine.py diubah (grayscale preprocessing),
karena embedding lama dihitung dengan pipeline RGB penuh -- gak
comparable lagi sama query baru.

Cara pakai:
    python manage.py regenerate_embeddings
    python manage.py regenerate_embeddings --dry-run   # cek dulu tanpa nyimpen
"""
from django.core.management.base import BaseCommand
from django.db import transaction
from penelusuran_fonetik.models import LogoDatabaseEntry
from penelusuran_fonetik.ml_engine import extract_embedding, embedding_to_str


class Command(BaseCommand):
    help = 'Regenerate embedding semua logo pakai pipeline ml_engine.py terbaru'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Hitung ulang tapi jangan disimpan, cuma tampilkan hasilnya',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        entries = LogoDatabaseEntry.objects.all()
        total = entries.count()

        self.stdout.write(f'Total logo yang akan diproses: {total}')
        if dry_run:
            self.stdout.write(self.style.WARNING('MODE DRY-RUN — tidak ada yang disimpan'))

        success_count = 0
        failed = []

        for idx, entry in enumerate(entries, 1):
            try:
                if not entry.logo_file:
                    failed.append((entry.nama_merek, 'file logo tidak ada'))
                    continue

                # baca ulang file gambar dari storage
                entry.logo_file.open('rb')
                img_bytes = entry.logo_file.read()
                entry.logo_file.close()

                new_emb = extract_embedding(img_bytes)

                if not dry_run:
                    entry.embedding = embedding_to_str(new_emb)
                    entry.save(update_fields=['embedding'])

                success_count += 1

            except Exception as e:
                failed.append((entry.nama_merek, str(e)))

            # progress tiap 50 logo
            if idx % 50 == 0 or idx == total:
                self.stdout.write(f'  [{idx}/{total}] diproses...')

        self.stdout.write(self.style.SUCCESS(
            f'\nSelesai. Berhasil: {success_count}/{total}'
        ))
        if failed:
            self.stdout.write(self.style.ERROR(f'Gagal: {len(failed)} logo'))
            for nama, err in failed[:20]:
                self.stdout.write(f'  - {nama}: {err}')

        if dry_run:
            self.stdout.write(self.style.WARNING(
                '\nIni dry-run, embedding BELUM disimpan. '
                'Jalankan tanpa --dry-run untuk apply beneran.'
            ))