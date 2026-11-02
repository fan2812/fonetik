import os
import re
import io
import time
import jellyfish

from PIL import Image
from django.conf import settings
from django.core.cache import cache
from django.db.models import Case, When, Value, IntegerField                                    # import cache Django
from .models import DataMerek, LogoDatabaseEntry
from .ml_engine import str_to_embedding, cosine_similarity
from .constants import (ALLOWED_LOGO_EXT, LOGO_SIM_THRESHOLD, MAX_LOGO_SIZE_MB, LOGO_NAME_RESCUE_THRESHOLD, LOGO_SIM_THRESHOLD_SIMBOL, LOGO_SIM_THRESHOLD_CAMPURAN, LOGO_SIM_THRESHOLD_TEKS, LOGO_PHONETIC_THRESHOLD_TEKS)

def normalize_phonetic(word):
    if not word:
        return ""
    word = str(word).upper().strip()
    word = re.sub(r'\s*[\+&\(\[].*', '', word)
    replacements = {
        #bahsa indonesia pola 2 huruf
        "DJ": "J", "TJ": "C", "OE": "U", "KH": "H", "GH": "H", "CH": "K", "X": "KS", "NK": "NG", "YI": "I", "AC": "ES",
        
        #bahasa indonesia pola 1 huruf
        "Y": "I", "P": "F", "V": "P", "Z": "S", "C": "K", "Q": "K", "W": "U", "J": "Y", "K": "C",
        
        #bahasa inggris pola 3 huruf atau lebih
        "THE": "DE", "AND": "END", "FOR": "FO", "WITH": "WIT", "OUGH": "AF", "TION": "SYN", "SION": "SYN", "IGH": "AY", "TIO": "SIO", 
        "EIGH": "AY", "OUGH": "AF", "OUGH": "OF", "OUGH": "AW", "OUGH": "OW", "OUGH": "OO", "OUGH": "UFF", "PHO": "FO", "ACE": "ES",
        "KEE": "KI", "EVE": "IV", "EAD": "ED", "EEL": "IL", "EAT": "IT", "ECK": "EK", "EEN": "IN", "EER": "ER", "TEA": "TI",
        
        #bahasa inggris pola 2 huruf
        "AN": "EN", "OF": "OV", "TO": "TU", "BY": "BAI", "ON": "ONN", "AT": "ET", "PH": "F", "EC": "ES", "EU": "U", "EE": "I", "ST": "S",
        
        #bahasa inggris pola 1 huruf
        "A": "E", "F": "P", "V": "P", "Z": "S", "C": "K", "Q": "K", "X": "KS", "W": "U", "J": "Y",
    }
    for old, new in replacements.items():
        word = word.replace(old, new)
    return word


def cari_fonetik(query, limit=50):
    """Logic inti pencarian fonetik. Dipanggil search_view (web) & api_search_fonetik (API)."""

    cache_key = f'fonetik_search_{query.lower().strip()}'               # key unik per query
    cached_result = cache.get(cache_key)                                # cek cache dulu
    if cached_result is not None:                                       # kalau ketemu, langsung return
        return cached_result                                            # skip seluruh scan database

    norm_query  = normalize_phonetic(query)
    results     = []

    for item in DataMerek.objects.all():
        norm_db     = normalize_phonetic(item.nama_merek)
        skor_awal   = jellyfish.jaro_winkler_similarity(norm_query, norm_db)
        jarak_huruf = jellyfish.levenshtein_distance(norm_query, norm_db)
        penalti     = jarak_huruf * 0.10
        skor_akhir  = max(0.0, skor_awal - penalti)

        if skor_akhir >= 0.75:
            results.append({
                'merek_input': query.upper(),
                'merek_db':    item.nama_merek,
                'similarity':  round(skor_akhir * 100, 2)
            })

    final_results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:limit]
    cache.set(cache_key, final_results, 300)                            # ★ PENTING — simpan ke cache, expire 5 menit
    return final_results


def validate_image_file(file_bytes, filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED_LOGO_EXT:
        return False, f'Format tidak didukung. Gunakan: {", ".join(sorted(ALLOWED_LOGO_EXT))}'

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_LOGO_SIZE_MB:
        return False, f'File terlalu besar ({size_mb:.1f}MB). Maks {MAX_LOGO_SIZE_MB}MB.'

    magic_signatures = [
        b'\x89PNG', b'\xff\xd8\xff', b'GIF8', b'BM',
        b'II*\x00', b'MM\x00*', b'RIFF',
    ]
    header   = file_bytes[:12]
    detected = any(header[:len(sig)] == sig for sig in magic_signatures)

    if file_bytes[:4] == b'RIFF' and file_bytes[8:12] != b'WEBP':
        detected = False

    if not detected:
        return False, 'File bukan gambar yang valid.'
    return True, ''


def cari_logo_mirip(query_emb, nama_merek_qry='', cache_key=None, jenis=None):
    if cache_key:
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

    norm_query = normalize_phonetic(nama_merek_qry) if nama_merek_qry else ''
    thr_vec, thr_fon = {
        'simbol':   (LOGO_SIM_THRESHOLD_SIMBOL,   LOGO_NAME_RESCUE_THRESHOLD),
        'campuran': (LOGO_SIM_THRESHOLD_CAMPURAN, LOGO_NAME_RESCUE_THRESHOLD),
        'teks':     (LOGO_SIM_THRESHOLD_TEKS,     LOGO_PHONETIC_THRESHOLD_TEKS),
    }.get(jenis, (LOGO_SIM_THRESHOLD, LOGO_NAME_RESCUE_THRESHOLD))   # jenis None -> perilaku lama
    results = []
    for entry in LogoDatabaseEntry.objects.exclude(embedding='').exclude(embedding='[]'):
        db_emb = str_to_embedding(entry.embedding)
        if not db_emb:
            continue
        score = cosine_similarity(query_emb, db_emb)
        lolos_vector = score >= thr_vec
        nama_cocok = False

        if norm_query:
            norm_db = normalize_phonetic(entry.nama_merek)
            skor_fonetik = jellyfish.jaro_winkler_similarity(norm_query, norm_db)
            skor_fonetik = max(0.0, skor_fonetik - jellyfish.levenshtein_distance(norm_query, norm_db) * 0.10)
            nama_cocok = skor_fonetik >= thr_fon

        if lolos_vector or nama_cocok:
            results.append({
                'id':           entry.id,
                'nama_merek':   entry.nama_merek,
                'logo_url':     entry.get_logo_url(),
                'similarity':   round(score * 100, 2),
                'lolos_vector': lolos_vector,
                'nama_cocok':   nama_cocok,
            })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    if cache_key:
        cache.set(cache_key, results, 300)
    return results

def cari_merek_autocomplete(query, limit=8):
    """Autocomplete nama merek dari DataMerek, buat dipakai di field live-search.
    Prioritas: exact match > startswith > contains biasa, baru alfabetis di tiap grup.
    Tanpa ini, nama yg exact match bisa 'ketutup' banyaknya hasil substring match lain."""
    query = (query or '').strip()
    if len(query) < 2:
        return []

    cache_key = f'merek_autocomplete_{query.lower()}'
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    matches = list(
        DataMerek.objects
        .filter(nama_merek__icontains=query)
        .annotate(priority=Case(
            When(nama_merek__iexact=query, then=Value(0)),
            When(nama_merek__istartswith=query, then=Value(1)),
            default=Value(2),
            output_field=IntegerField(),
        ))
        .order_by('priority', 'nama_merek')
        .values_list('nama_merek', flat=True)[:limit]
    )
    cache.set(cache_key, matches, 300)
    return matches


def cek_kecocokan_nama_dan_ocr(nama_merek_qry, teks_ocr, min_score=0.65):
    """True kalau nama yang diketik user masuk akal dibanding teks yang kebaca di gambar
    (dicek per-kata, diambil skor terbaiknya). False kalau kemungkinan besar user salah
    ketik / lupa ganti field nama merek pas ganti gambar query.
    Kalau salah satu input kosong (nama gak diisi, atau OCR gagal baca apapun),
    dianggap aman dan gak nge-block — bukan tanggung jawab fungsi ini buat maksa OCR
    selalu berhasil."""
    if not nama_merek_qry or not teks_ocr:
        return True

    norm_input = normalize_phonetic(nama_merek_qry)
    kata_ocr   = teks_ocr.upper().split()

    skor_terbaik = 0.0
    for kata in kata_ocr:
        norm_kata = normalize_phonetic(kata)
        skor = jellyfish.jaro_winkler_similarity(norm_input, norm_kata)
        skor_terbaik = max(skor_terbaik, skor)

    return skor_terbaik >= min_score


def filter_by_phonetic(nama_merek_query, vector_hits, min_score=0.75):
    """
    Cascade step 2: filter hasil vector similarity (dari cari_logo_mirip)
    berdasarkan kedekatan fonetik nama merek query vs nama tiap kandidat.
    Kandidat yang fonetiknya jauh dari nama_merek_query akan didrop,
    walau skor vector-nya tinggi. Kalau nama_merek_query kosong, cascade
    di-skip dan vector_hits dikembalikan apa adanya.
    """
    nama_merek_query = (nama_merek_query or '').strip()
    if not nama_merek_query:
        return vector_hits

    norm_query = normalize_phonetic(nama_merek_query)
    filtered = []
    for hit in vector_hits:
        norm_hit    = normalize_phonetic(hit['nama_merek'])
        skor_awal   = jellyfish.jaro_winkler_similarity(norm_query, norm_hit)
        jarak_huruf = jellyfish.levenshtein_distance(norm_query, norm_hit)
        skor_fonetik = max(0.0, skor_awal - jarak_huruf * 0.10)

        if skor_fonetik >= min_score:
            hit = {**hit, 'phonetic_score': round(skor_fonetik * 100, 2)}
            filtered.append(hit)
    return filtered


def cleanup_tmp_files():
    tmp_dir = os.path.join(settings.MEDIA_ROOT, 'tmp')
    if not os.path.exists(tmp_dir):
        return
    now = time.time()
    for fname in os.listdir(tmp_dir):
        fpath = os.path.join(tmp_dir, fname)
        try:
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > 3600:
                os.remove(fpath)
        except Exception:
            pass
        
def optimize_logo_image(img_bytes, source_name, max_dim=1200, quality=85):
    """
    Konversi gambar upload ke WebP dan downscale kalau kegedean, sebelum
    disimpan ke folder media/logos/. Mencegah asset folder numpuk file
    mentah beraneka format/ukuran. Return (webp_bytes, new_filename).
    """
    img = Image.open(io.BytesIO(img_bytes))

    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    # WebP dukung transparansi -- convert ke RGBA kalau ada alpha,
    # RGB kalau nggak, daripada maksa semua RGB dan ngilangin transparansi.
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')
    else:
        img = img.convert('RGB')

    buffer = io.BytesIO()
    img.save(buffer, format='WEBP', quality=quality)
    buffer.seek(0)

    stem = source_name.rsplit('.', 1)[0]
    return buffer.getvalue(), f'{stem}.webp'