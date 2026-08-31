# ================================================================
# SKRIP A/B TESTING — Padding-only vs Background-removal
# Jalankan via: python manage.py shell < ab_test_bg_removal.py
# atau paste manual ke `python manage.py shell`
#
# Isi PASANGAN_LOGO di bawah dengan path file logo kamu.
# Pasangan "seharusnya_mirip" untuk cek apakah metode berhasil MENDETEKSI kemiripan
# Pasangan "seharusnya_beda" untuk cek apakah metode SALAH mendeteksi mirip (false positive)
# ================================================================

from PIL import Image
from penelusuran_fonetik.ml_engine import (
    extract_embedding, cosine_similarity, _resize_with_padding, _load_model
)

# rembg opsional — kalau belum diinstall, bagian bg-removal di-skip otomatis
try:
    from rembg import remove as _bg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("[!] rembg belum terinstall — hanya jalankan metode padding-only")


def embedding_padding_only(img: Image.Image) -> list:
    """Metode B: padding aspect-ratio-safe saja, tanpa background removal."""
    import torch
    model, transform = _load_model()
    padded = _resize_with_padding(img.convert('RGB'), 224)
    tensor = transform(padded).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor).squeeze()
    norm = emb.norm()
    if norm > 0:
        emb = emb / norm
    return emb.numpy().tolist()


def embedding_with_bg_removal(img: Image.Image) -> list:
    """Metode A: background removal + padding."""
    import torch
    model, transform = _load_model()
    img_no_bg = _bg_remove(img.convert('RGBA'))
    canvas = Image.new('RGBA', img_no_bg.size, (255, 255, 255, 255))
    canvas.paste(img_no_bg, (0, 0), img_no_bg)
    stripped = canvas.convert('RGB')
    padded = _resize_with_padding(stripped, 224)
    tensor = transform(padded).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor).squeeze()
    norm = emb.norm()
    if norm > 0:
        emb = emb / norm
    return emb.numpy().tolist()


# ================================================================
# ISI PASANGAN LOGO DI SINI — ganti path sesuai file kamu
# ================================================================
PASANGAN_SEHARUSNYA_MIRIP = [
    (r'C:\fonetik\media\logos\logo.png', r'C:\Users\taufa\Downloads\OIP (4).webp', 'BINUS v1 vs v2'),
    (r'C:\fonetik\media\logos\OIP_1.webp', r'C:\Users\taufa\Downloads\Chatime-Emblem.png', 'chatime v1 vs v2'),
    (r'C:\fonetik\media\logos\image_uIUBBoZ.png', r'C:\Users\taufa\Downloads\garuda-indonesia-logo-png-transparent-svg-vector-20.png', 'garuda v1 vs v2'),
    (r'C:\fonetik\media\logos\image_GIMle4k.png', r'C:\Users\taufa\Downloads\cupbop-logo-png_seeklogo-649295.png', 'cupbop v1 vs v2'),
    (r'C:\fonetik\media\logos\image_0azf2g7.png', r'C:\Users\taufa\Downloads\2206712.webp', 'gojek v1 vs v2'),
    (r'C:\fonetik\media\logos\R_3.png', r'C:\Users\taufa\Downloads\logo-teazzi-599x558.jpeg', 'teazzi v1 vs v2'),
    (r'C:\fonetik\media\logos\image_lMby14H.png', r'C:\Users\taufa\Downloads\mobil-1-logo-png-transparent-png-17.png', 'mobil1 v1 vs v2'),
    (r'C:\fonetik\media\logos\image_Wwm8Gqe.png', r'C:\Users\taufa\Downloads\shell-logo-1971.webp', 'shell v1 vs v2'),
    (r'C:\fonetik\media\logos\image_hEOtXHi.png', r'C:\Users\taufa\Downloads\Kraft-Foods-Logo-2012.png', 'kraft v1 vs v2'),
    (r'C:\fonetik\media\logos\image_x6jZ5HV.png', r'"C:\Users\taufa\Downloads\Xiaomi-Logo-2019-present.jpg"', 'xiaomi v1 vs v2'),
    (r'C:\fonetik\media\logos\OIP_6.webp', r'"C:\Users\taufa\Downloads\Samsung_old_logo_before_year_2015.svg.webp"', 'samsung v1 vs v2'),
    (r'C:\fonetik\media\logos\image_6amuwFf.png', r'C:\Users\taufa\Downloads\Burger-King-Logo-1969-1994.png', 'burgerking v1 vs v2'),    
    # tambah pasangan lain di sini...
]

PASANGAN_SEHARUSNYA_BEDA = [
    (r'C:\path\ke\teazzi.png', r'C:\path\ke\logo_lain.png', 'Teazzi vs Logo Lain'),
    # tambah pasangan lain di sini...
]


def run_test():
    print("\n" + "="*70)
    print("HASIL A/B TESTING")
    print("="*70)

    hasil = []

    def proses_pasangan(pasangan, label_kategori):
        for path1, path2, nama in pasangan:
            img1 = Image.open(path1)
            img2 = Image.open(path2)

            score_b = cosine_similarity(
                embedding_padding_only(img1),
                embedding_padding_only(img2)
            )

            score_a = None
            if REMBG_AVAILABLE:
                score_a = cosine_similarity(
                    embedding_with_bg_removal(img1),
                    embedding_with_bg_removal(img2)
                )

            print(f"\n[{label_kategori}] {nama}")
            print(f"  Padding-only      : {score_b*100:.2f}%")
            if score_a is not None:
                print(f"  Dengan bg-removal : {score_a*100:.2f}%")
                print(f"  Selisih           : {(score_a-score_b)*100:+.2f}%")

            hasil.append({
                'nama': nama,
                'kategori': label_kategori,
                'padding_only': round(score_b*100, 2),
                'bg_removal': round(score_a*100, 2) if score_a is not None else None,
            })

    proses_pasangan(PASANGAN_SEHARUSNYA_MIRIP, 'SEHARUSNYA MIRIP')
    proses_pasangan(PASANGAN_SEHARUSNYA_BEDA, 'SEHARUSNYA BEDA')

    print("\n" + "="*70)
    print("RINGKASAN (copy hasil ini untuk diberikan ke Claude buat divisualisasikan)")
    print("="*70)
    import json
    print(json.dumps(hasil, indent=2))

    return hasil


if __name__ == '__main__':
    run_test()