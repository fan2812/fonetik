import os
import re
import time
import jellyfish

from django.conf import settings
from django.core.cache import cache                                    # import cache Django
from .models import DataMerek, LogoDatabaseEntry
from .ml_engine import str_to_embedding, cosine_similarity
from .constants import ALLOWED_LOGO_EXT, LOGO_SIM_THRESHOLD, MAX_LOGO_SIZE_MB

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
        "KEE": "KI", "EVE": "IV", "EAD": "ED", "EEL": "IL", "EAT": "IT", "ECK": "EK", "EEN": "IN", "EER": "ER",
        
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


def cari_logo_mirip(query_emb, cache_key=None):
    """Logic inti similarity logo. Dipanggil logo_search_view (web) & api_search_logo (API)."""
    if cache_key:
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result
    
    results = []
    for entry in LogoDatabaseEntry.objects.exclude(embedding='').exclude(embedding='[]'):
        db_emb = str_to_embedding(entry.embedding)
        if not db_emb:
            continue
        score = cosine_similarity(query_emb, db_emb)
        if score >= LOGO_SIM_THRESHOLD:
            results.append({
                'nama_merek': entry.nama_merek,
                'logo_url':   entry.get_logo_url(),
                'similarity': round(score * 100, 2),
            })
    results.sort(key=lambda x: x['similarity'], reverse=True)
    if cache_key:
        cache.set(cache_key, results, 300)  # Cache for 5 minutes
    return results

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