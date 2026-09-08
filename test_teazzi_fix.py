# test_teazzi_fix.py
import os
import datetime
import numpy as np
from your_app.models import LogoDatabaseEntry  # Sesuaikan nama model DB kamu
from ml_engine import extract_embedding         # Sesuaikan fungsi ekstraksi kamu
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'nama_project_kamu.settings') # Ganti nama_project_kamu
django.setup()

# Kode sisanya tetap sama...

# --- LANGKAH 1: Cek Timeline Mismatch ---
entry = LogoDatabaseEntry.objects.filter(name__icontains='teazzi').first()

if not entry:
    print("Entry 'teazzi' tidak ditemukan di database!")
else:
    print("=== LANGKAH 1: CEK TIMELINE ===")
    print(f"Updated_at Teazzi di DB : {entry.updated_at}")
    
    # Cek waktu modifikasi file ml_engine.py
    ml_engine_mtime = os.path.getmtime('ml_engine.py')
    ml_engine_date = datetime.datetime.fromtimestamp(ml_engine_mtime)
    print(f"Terakhir ml_engine.py diubah: {ml_engine_date}")
    
    if entry.updated_at < ml_engine_date:
        print("-> CONFIRMED: Embedding Teazzi di DB dibuat SEBELUM perubahan ml_engine.py!\n")
    else:
        print("-> Timestamp DB lebih baru atau sama dengan skrip.\n")

    # --- LANGKAH 2: Single-Entry Regenerate & Re-test ---
    print("=== LANGKAH 2: REGENERATE & TEST SIMILARITY ===")
    
    # Path gambar query teazzi yang kamu pakai buat search
    query_image_path = "path/to/teazzi_white_bg.png"  # Ganti dengan path gambar pencarian kamu
    
    # 1. Regenerate embedding entry DB menggunakan pipeline ml_engine.py versi terbaru
    print("Mengenerate ulang embedding Teazzi di DB...")
    updated_db_vector = extract_embedding(entry.image_path)
    entry.embedding = updated_db_vector
    entry.save()
    print("Embedding Teazzi di DB berhasil diperbarui!")

    # 2. Extract embedding gambar search (query)
    query_vector = extract_embedding(query_image_path)

    # 3. Hitung Cosine Similarity
    def cosine_similarity(v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    score = cosine_similarity(updated_db_vector, query_vector)
    print(f"\nSkor Similarity Terbaru: {score * 100:.2f}%")