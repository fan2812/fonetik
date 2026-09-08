# penelusuran_fonetik/constants.py
"""
Konfigurasi terpusat. Jangan hardcode angka/string ini di views.py,
services.py, atau api_views.py — import dari sini aja.
"""

# ================================================================
# DIPAKE DI views.py (upload_database)
# ================================================================
ALLOWED_CSV_EXT = {'csv', 'xlsx', 'xls'}
MAX_CSV_SIZE_MB = 50
CSV_ENCODINGS_TRY = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']

# ================================================================
# DIPAKE DI services.py (validate_image_file, cari_logo_mirip)
# ================================================================
ALLOWED_LOGO_EXT = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tif', 'tiff'}
MAX_LOGO_SIZE_MB = 10
LOGO_SIM_THRESHOLD = 0.75

# ================================================================
# import di services.py (cari_logo_mirip)
# ================================================================
FONETIK_SIMILARITY_THRESHOLD = 0.35
LOGO_NAME_RESCUE_THRESHOLD = 0.90
FONETIK_PENALTI_PER_HURUF = 0.10
DUPLICATE_LOGO_THRESHOLD = 0.90

PHONETIC_REPLACEMENTS = {
    "DJ": "J", "TJ": "C", "OE": "U", "Y": "I",
    "F": "P", "V": "P", "Z": "S", "KH": "H", "GH": "H", "CH": "K", "C": "K", "Q": "K",
    "X": "KS", "W": "U", "J": "Y", "NK": "NG", "YI": "I"
}

MAGIC_SIGNATURES = [
    b'\x89PNG', b'\xff\xd8\xff', b'GIF8', b'BM',
    b'II*\x00', b'MM\x00*', b'RIFF',
]

RATE_LIMIT_LOGO_SEARCH_MAX = 10
RATE_LIMIT_LOGO_SEARCH_WINDOW_SECONDS = 60
CACHE_KEY_PREFIX_RATE_LIMIT = 'ratelimit_logo_'
MAX_URL_IMAGE_MB = 5
TMP_FILE_MAX_AGE_SECONDS = 3600