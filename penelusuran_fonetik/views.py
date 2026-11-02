import hashlib
import io
import os
import time
import requests
import pandas as pd
import openpyxl
import ipaddress
import socket
import re
from urllib.parse import urlparse, urljoin, unquote
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from . import models as _models

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import logout, login
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.db import transaction
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.core.cache import cache
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.conf import settings

from .ml_engine import extract_embedding, extract_embedding_and_type, embedding_to_str, extract_text_from_logo
from .services import (
    cari_fonetik, validate_image_file, cari_logo_mirip, cleanup_tmp_files,
    cari_merek_autocomplete, cek_kecocokan_nama_dan_ocr, optimize_logo_image,
)
from .models import DatabaseChangeLog, DataMerek, LogoDatabaseEntry
from .forms import UploadCSVForm
from .constants import (
    ALLOWED_CSV_EXT, DUPLICATE_LOGO_THRESHOLD, MAX_CSV_SIZE_MB, CSV_ENCODINGS_TRY,
    RATE_LIMIT_LOGO_SEARCH_MAX, RATE_LIMIT_LOGO_SEARCH_WINDOW_SECONDS, CACHE_KEY_PREFIX_RATE_LIMIT,
)

MANUAL_MIN_ROWS = 21   # Di bawah ini wajib manual, bukan file
MAX_URL_IMAGE_MB = 5
MAX_URL_REDIRECTS = 3
URL_FETCH_DEADLINE_SECONDS = 15


def is_staff(user):
    return user.is_authenticated and user.is_staff

def is_admin(user):
    return user.is_authenticated and user.is_staff


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def _is_public_host(hostname):
    """True hanya kalau semua IP dari hostname adalah alamat publik internet."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, UnicodeError):
        return False
    if not infos:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split('%')[0])
        if ip.version == 6 and ip.ipv4_mapped:      # buka bungkus ::ffff:127.0.0.1
            ip = ip.ipv4_mapped
        if (not ip.is_global) or ip.is_multicast or ip.is_unspecified:
            return False
    return True


def _url_aman(parsed):
    """True kalau URL: http/https, tanpa kredensial, port 80/443, dan host publik."""
    if parsed.scheme not in ('http', 'https'):
        return False
    if parsed.username or parsed.password:
        return False
    try:
        port = parsed.port
    except ValueError:
        return False
    if port not in (None, 80, 443):
        return False
    return bool(parsed.hostname) and _is_public_host(parsed.hostname)


def _fetch_image_from_url(url):
    """Ambil gambar dari URL publik. Return (img_bytes, filename, error)."""
    pesan_gagal = 'Gagal mengambil gambar dari URL tersebut.'
    limit       = MAX_URL_IMAGE_MB * 1024 * 1024
    deadline    = time.monotonic() + URL_FETCH_DEADLINE_SECONDS
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # Ikuti redirect secara manual; setiap lompatan dicek ulang seperti URL pertama.
    current, resp = url, None
    for _ in range(MAX_URL_REDIRECTS + 1):
        parsed = urlparse(current)
        if not _url_aman(parsed):
            return None, None, 'URL tidak valid.'
        try:
            resp = requests.get(current, timeout=10, stream=True,
                                headers=headers, allow_redirects=False)
        except requests.RequestException:
            return None, None, pesan_gagal
        if resp.is_redirect:
            lokasi = resp.headers.get('Location')
            resp.close()
            resp = None
            if not lokasi:
                return None, None, pesan_gagal
            current = urljoin(current, lokasi)
            continue
        break
    else:
        return None, None, pesan_gagal          # terlalu banyak redirect

    try:
        if resp.status_code != 200:
            return None, None, pesan_gagal
        if not resp.headers.get('Content-Type', '').startswith('image/'):
            return None, None, pesan_gagal

        content_length = resp.headers.get('Content-Length')
        if content_length and content_length.isdigit() and int(content_length) > limit:
            return None, None, f'Gambar terlalu besar. Maks {MAX_URL_IMAGE_MB}MB.'

        buf = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            buf.extend(chunk)
            if len(buf) > limit:
                return None, None, f'Gambar terlalu besar. Maks {MAX_URL_IMAGE_MB}MB.'
            if time.monotonic() > deadline:
                return None, None, pesan_gagal
        img_bytes = bytes(buf)
    except requests.RequestException:
        return None, None, pesan_gagal
    finally:
        resp.close()

    filename = unquote(parsed.path.rsplit('/', 1)[-1])
    filename = re.sub(r'[^A-Za-z0-9._-]', '_', filename).strip('._')[:100] or 'logo_from_url'
    if '.' not in filename:
        filename += '.jpg'

    return img_bytes, filename, None

def _baca_media_lokal(url, request):
    """Kalau url menunjuk ke file MEDIA di situs ini sendiri (mis. logo dari database yang tampil
    di halaman), baca langsung dari storage tanpa request HTTP. Return (bytes, nama_file) atau None."""
    p = urlparse(url)
    if p.scheme not in ('', 'http', 'https'):
        return None
    if p.netloc and p.netloc != request.get_host():
        return None
    path = unquote(p.path)
    if not path.startswith(settings.MEDIA_URL) or '..' in path or '\\' in path:
        return None
    rel = path[len(settings.MEDIA_URL):]
    if not rel:
        return None
    try:
        with default_storage.open(rel, 'rb') as f:
            return f.read(), os.path.basename(rel)
    except Exception:
        return None


def _rate_limited(request):
    """Return 0 kalau pencarian boleh lanjut. Kalau sudah melewati batas, return sisa detik
    sampai jendela waktu direset (dipakai untuk hitung mundur di halaman)."""
    ip     = request.META.get('REMOTE_ADDR', 'unknown')
    key    = f'{CACHE_KEY_PREFIX_RATE_LIMIT}{ip}'
    now    = time.time()
    window = RATE_LIMIT_LOGO_SEARCH_WINDOW_SECONDS

    data = cache.get(key)                       # (jumlah_request, waktu_mulai_jendela)
    if not data or now - data[1] >= window:
        cache.set(key, (1, now), window)        # jendela baru
        return 0

    jumlah = data[0] + 1
    sisa   = max(1, int(window - (now - data[1])) + 1)
    cache.set(key, (jumlah, data[1]), sisa)     # masa berlaku tetap mengikuti jendela awal
    return sisa if jumlah > RATE_LIMIT_LOGO_SEARCH_MAX else 0

def _catat_log_logo(request, status, notes):
    """Catat ke DatabaseChangeLog dan kembalikan field log_* untuk baris log di halaman."""
    DatabaseChangeLog.objects.create(
        admin_user = request.user,
        filename = 'Manual Edit',
        fileformat = 'manual',
        rows_before = LogoDatabaseEntry.objects.count(),
        rows_after = LogoDatabaseEntry.objects.count(),
        status = status,
        notes = notes,
    )
    now = timezone.localtime()
    return {
        'log_time': now.strftime('%d %b %Y'), 'log_time_detail': now.strftime('%H:%M:%S'),
        'log_admin': request.user.username, 'log_filename': 'Manual Edit',
        'log_notes': notes, 'log_status': status,
    }
    
def _tolak_edit_logo(request, nama_lama, pesan):
    """Edit ditolak: catat sebagai 'failed' di log, lalu kembalikan error ke halaman."""
    log = _catat_log_logo(request, 'failed', f'Gagal edit logo {nama_lama}: {pesan}')
    return JsonResponse({'status': 'error', 'message': pesan, **log}, status=400)
    
# ================================================================
# 1. PENCARIAN FONETIK (publik)
# ================================================================

def search_view(request):
    query   = request.GET.get('q', '').strip()
    results = cari_fonetik(query) if query else []

    context = {
        'query':               query,
        'results':             results,
        'total_database':     DataMerek.objects.count(),
        'kemiripan_ditemukan': len(results),
    }
    return render(request, 'penelusuran_fonetik/result.html', context)

# ================================================================
# 2. UPDATE DB MEREK (admin)
# ================================================================

@login_required
@user_passes_test(is_staff, login_url='/login/')
def upload_database(request):

    # ── A. Tambah merek manual (satu per satu) ─────────────────
    if request.method == 'POST' and request.POST.get('action') == 'add_manual':
        nama_merek = request.POST.get('nama_merek', '').strip().upper()
        if not nama_merek:
            return JsonResponse(
                {'status': 'error', 'message': 'Nama merek tidak boleh kosong.'},
                status=400
            )

        rows_before = DataMerek.objects.count()
        _models._disable_signals = True
        try:
            obj, created = DataMerek.objects.get_or_create(
                nama_merek=nama_merek,
                defaults={'uploaded_by': request.user}
            )
        finally:
            _models._disable_signals = False
        rows_after = DataMerek.objects.count()

        DatabaseChangeLog.objects.create(
            admin_user  = request.user,
            filename    = 'Manual Input',
            fileformat  = 'manual',
            rows_before = rows_before,
            rows_after  = rows_after,
            status      = 'success',
            notes       = (
                f'Tambah manual: {nama_merek}' if created
                else f'Merek "{nama_merek}" sudah ada di database.'
            ),
        )

        cache.clear()

        if created:
            return JsonResponse({
                'status':  'success',
                'message': f'Merek "{nama_merek}" berhasil ditambahkan.',
                'total':   rows_after,
            })
        return JsonResponse({
            'status':  'warning',
            'message': f'Merek "{nama_merek}" sudah ada di database.',
            'total':   rows_after,
        })

    # ── B. Upload file (Excel/CSV) ──────────────────────────────
    if request.method == 'POST' and request.FILES:
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return JsonResponse(
                {'status': 'error', 'message': 'Tidak ada file yang diunggah'},
                status=400
            )

        filename  = uploaded_file.name
        extension = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

        if extension not in ALLOWED_CSV_EXT:
            return JsonResponse(
                {'status': 'error', 'message': 'Format tidak didukung. Gunakan .csv atau .xlsx'},
                status=400
            )

        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > MAX_CSV_SIZE_MB:
            return JsonResponse(
                {'status': 'error', 'message': f'File terlalu besar. Maks {MAX_CSV_SIZE_MB}MB.'},
                status=400
            )

        rows_before = DataMerek.objects.count()

        try:
            if extension == 'csv':
                content = uploaded_file.read()
                decoded_file = None
                for encoding in CSV_ENCODINGS_TRY:
                    try:
                        decoded_file = content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                if decoded_file is None:
                    return JsonResponse({
                        'status':  'error',
                        'message': 'Encoding file tidak dikenali. Simpan ulang sebagai UTF-8.'
                    }, status=400)

                io_string = io.StringIO(decoded_file)
                df        = pd.read_csv(io_string, sep=None, engine='python')

                if len(df) < MANUAL_MIN_ROWS:
                    return JsonResponse({
                        'status':  'error',
                        'message': (
                            f'File hanya berisi {len(df)} merek. '
                            f'Upload file minimal {MANUAL_MIN_ROWS} merek. '
                            f'Untuk jumlah lebih sedikit, gunakan tambah manual.'
                        )
                    }, status=400)

                target_col = df.columns[0]
                for col in df.columns:
                    if 'merek' in str(col).lower():
                        target_col = col
                        break

                with transaction.atomic():
                    raw_values   = df[target_col].dropna().astype(str)
                    clean_values = {val.strip().upper() for val in raw_values if val.strip()}
                    _models._disable_signals = True
                    try:
                        for merek_nama in clean_values:
                            DataMerek.objects.get_or_create(
                                nama_merek=merek_nama,
                                defaults={'uploaded_by': request.user}
                            )
                    finally:
                        _models._disable_signals = False

            else:  # Excel
                wb = openpyxl.load_workbook(uploaded_file, read_only=True)
                ws = wb.active

                headers = [
                    str(cell.value).strip().lower() if cell.value else ''
                    for cell in ws[1]
                ]
                merek_col_idx = 0
                for i, h in enumerate(headers):
                    if 'merek' in h:
                        merek_col_idx = i
                        break

                data_rows = list(ws.iter_rows(min_row=2, values_only=True))
                data_rows = [r for r in data_rows if len(r) > merek_col_idx and r[merek_col_idx]]

                if len(data_rows) < MANUAL_MIN_ROWS:
                    wb.close()
                    return JsonResponse({
                        'status':  'error',
                        'message': (
                            f'File hanya berisi {len(data_rows)} merek. '
                            f'Upload file minimal {MANUAL_MIN_ROWS} merek. '
                            f'Untuk jumlah lebih sedikit, gunakan tambah manual.'
                        )
                    }, status=400)

                with transaction.atomic():
                    _models._disable_signals = True
                    try:
                        for row in data_rows:
                            val = row[merek_col_idx]
                            if val:
                                nama = str(val).strip().upper()
                                if nama:
                                    DataMerek.objects.get_or_create(
                                        nama_merek=nama,
                                        defaults={'uploaded_by': request.user}
                                    )
                    finally:
                        _models._disable_signals = False
                wb.close()

            rows_after  = DataMerek.objects.count()
            added_count = rows_after - rows_before

            DatabaseChangeLog.objects.create(
                admin_user  = request.user,
                filename    = filename,
                fileformat  = extension,
                rows_before = rows_before,
                rows_after  = rows_after,
                status      = 'success',
                notes       = f'Upload berhasil via Web. Ditambah {added_count} data baru.'
            )

            cache.clear()

            return JsonResponse({
                'status':  'success',
                'message': f'Berhasil! {added_count} data baru ditambahkan. Total: {rows_after}.'
            })

        except Exception as e:
            DatabaseChangeLog.objects.create(
                admin_user  = request.user,
                filename    = filename,
                fileformat  = extension,
                rows_before = rows_before,
                rows_after  = rows_before,
                status      = 'failed',
                notes       = str(e)
            )
            return JsonResponse({
                'status':  'error',
                'message': 'Gagal memproses file. Pastikan format kolom sesuai.'
            }, status=500)

    return render(request, 'upload.html')


# ================================================================
# 3. LOG DATABASE (admin)
# ================================================================
@login_required
@user_passes_test(is_staff)
def database_log_view(request):
    logs         = DatabaseChangeLog.objects.select_related('admin_user').all().order_by('-timestamp')
    total_brands = DataMerek.objects.count()
    total_logos  = LogoDatabaseEntry.objects.count()
    total_logs   = logs.count()
    from_page    = request.GET.get('from', 'merek')

    merek_query      = request.GET.get('mq', '').strip()
    merek_list       = None
    logos_list       = None
    merek_tanpa_logo = None

    if from_page == 'merek':
        merek_list = DataMerek.objects.none()
        if merek_query:
            merek_list = DataMerek.objects.filter(nama_merek__icontains=merek_query).order_by('-created_at')[:100]
    elif from_page == 'logo':
        logos_list       = LogoDatabaseEntry.objects.none()
        merek_tanpa_logo = DataMerek.objects.none()
        if merek_query:
            logos_list = LogoDatabaseEntry.objects.filter(nama_merek__icontains=merek_query).order_by('-created_at')[:100]
            nama_ada_logo = LogoDatabaseEntry.objects.values_list('nama_merek', flat=True)
            merek_tanpa_logo = DataMerek.objects.filter(
                nama_merek__icontains=merek_query
            ).exclude(nama_merek__in=nama_ada_logo).order_by('-created_at')[:100]

    context = {
        'logs':             logs,
        'total_brands':     total_brands,
        'total_logos':      total_logos,
        'total_logs':       total_logs,
        'from_page':        from_page,
        'merek_query':      merek_query,
        'merek_list':       merek_list,
        'logos_list':       logos_list,
        'merek_tanpa_logo': merek_tanpa_logo,
    }
    return render(request, 'log.html', context)


@login_required
@user_passes_test(is_staff)
@require_http_methods(["POST"])
def merek_delete_view(request, merek_id):
    try:
        merek = DataMerek.objects.get(id=merek_id)
        nama = merek.nama_merek
        rows_before = DataMerek.objects.count()

        _models._disable_signals = True
        try:
            merek.delete()
        finally:
            _models._disable_signals = False

        rows_after = DataMerek.objects.count()

        DatabaseChangeLog.objects.create(
            admin_user  = request.user,
            filename    = 'Manual Delete',
            fileformat  = 'manual',
            rows_before = rows_before,
            rows_after  = rows_after,
            status      = 'deleted',
            notes       = f'Hapus manual: {nama}',
        )

        cache.clear()

        now = timezone.localtime()
        return JsonResponse({
            'status': 'ok', 'message': f'Merek "{nama}" dihapus.',
            'log_time': now.strftime('%d %b %Y'), 'log_time_detail': now.strftime('%H:%M:%S'),
            'log_admin': request.user.username, 'log_filename': 'Manual Delete',
            'log_notes': f'Hapus manual: {nama}', 'log_status': 'deleted',
        })
    except DataMerek.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Merek tidak ditemukan.'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@login_required
@user_passes_test(is_staff)
@require_http_methods(["POST"])
def merek_edit_view(request, merek_id):
    try:
        merek = DataMerek.objects.get(id=merek_id)
        nama_lama = merek.nama_merek
        nama_baru = request.POST.get('nama_merek', '').strip().upper()

        if not nama_baru:
            return JsonResponse({'status': 'error', 'message': 'Nama merek tidak boleh kosong.'}, status=400)
        if nama_baru == nama_lama:
            return JsonResponse({'status': 'warning', 'message': 'Tidak ada perubahan.'}, status=200)
        if DataMerek.objects.filter(nama_merek=nama_baru).exclude(id=merek_id).exists():
            return JsonResponse({'status': 'error', 'message': f'Merek "{nama_baru}" sudah ada di database.'}, status=400)

        rows_before = DataMerek.objects.count()

        merek.nama_merek = nama_baru
        merek.save()

        rows_after = DataMerek.objects.count()

        DatabaseChangeLog.objects.create(
            admin_user  = request.user,
            filename    = 'Manual Edit',
            fileformat  = 'manual',
            rows_before = rows_before,
            rows_after  = rows_after,
            status      = 'success',
            notes       = f'Edit manual: {nama_lama} -> {nama_baru}',
        )

        cache.clear()

        now = timezone.localtime()
        return JsonResponse({
            'status': 'ok', 'message': f'Berhasil diubah menjadi "{nama_baru}".', 'nama_baru': nama_baru,
            'log_time': now.strftime('%d %b %Y'), 'log_time_detail': now.strftime('%H:%M:%S'),
            'log_admin': request.user.username, 'log_filename': 'Manual Edit',
            'log_notes': f'Edit manual: {nama_lama} -> {nama_baru}', 'log_status': 'success',
        })
    except DataMerek.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Merek tidak ditemukan.'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


# ================================================================
# 4. AUTH VIEWS
# ================================================================

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Selamat datang, {user.username}!')
            return redirect('search_view')
    else:
        form = UserCreationForm()
    return render(request, 'penelusuran_fonetik/register.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('search_view')


# ================================================================
# 5. SESSION & STAFF CHECK ENDPOINTS
# ================================================================

def session_check_view(request):
    if request.user.is_authenticated:
        return JsonResponse({'status': 'ok'}, status=200)
    return JsonResponse({'status': 'expired'}, status=401)


def staff_check_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({'is_staff': False})
    request.user.refresh_from_db()
    return JsonResponse({'is_staff': request.user.is_staff})


# ================================================================
# 6. LOGO SIMILARITY VIEWS (ML — ResNet50)
# ================================================================
def logo_search_view(request):
    cleanup_tmp_files()
    
    sisa_tunggu = _rate_limited(request) if request.method == 'POST' else 0
    if sisa_tunggu:
        return render(request, 'logo_search.html', {
            'error': f'Terlalu banyak pencarian. Coba lagi dalam {sisa_tunggu} detik.',
            'rate_limit_sisa': sisa_tunggu,
            'total_logo_db': LogoDatabaseEntry.objects.count(),
            'hasil_visual': [], 'hasil_rescue': [], 'kemiripan_ditemukan': 0,
            'nama_merek_qry': '', 'query_logo': None, 'tmp_logo_path': '',
        }, status=429)

    results        = []
    hasil_visual   = []
    hasil_rescue   = []
    query_logo     = None
    tmp_logo_path  = None
    error          = None
    nama_merek_qry = ''
    total_logo_db  = LogoDatabaseEntry.objects.count()

    if request.method == 'POST':
        uploaded       = request.FILES.get('logo')
        logo_url       = request.POST.get('logo_url', '').strip()
        tmp_logo_path  = request.POST.get('tmp_logo_path', '').strip()
        nama_merek_qry = request.POST.get('nama_merek', '').strip()
        
        img_bytes = None
        source_name = None

        if tmp_logo_path:
            # hanya terima file sementara buatan view ini (cegah path traversal)
            if tmp_logo_path.startswith('tmp/') and '..' not in tmp_logo_path:
                try:
                    with default_storage.open(tmp_logo_path, 'rb') as f:
                        img_bytes = f.read()
                    # buang awalan 'query_<angka>_' yang menumpuk tiap file disimpan ulang
                    source_name = re.sub(r'^(?:query_\d+_)+', '', os.path.basename(tmp_logo_path)) or 'logo.png'
                except Exception as e:
                    print(f'[WARN] gagal baca tmp {tmp_logo_path!r}: {type(e).__name__}: {e}')
                    error = 'Gambar sebelumnya sudah tidak tersedia, silakan upload ulang.'
            else:
                tmp_logo_path = ''
                error = 'Gambar sebelumnya tidak valid, silakan upload ulang.'
        elif uploaded:
            img_bytes   = uploaded.read()
            source_name = uploaded.name
        elif logo_url:
            lokal = _baca_media_lokal(logo_url, request)
            if lokal:
                img_bytes, source_name = lokal
            else:
                img_bytes, source_name, fetch_err = _fetch_image_from_url(logo_url)
                if fetch_err:
                    error = fetch_err
        else:
            error = 'Tidak ada file atau URL yang diberikan.'

        if img_bytes and not error:
            valid, err_msg = validate_image_file(img_bytes, source_name)

            if not valid:
                error = err_msg
            else:
                try:
                    tmp_path   = f'tmp/query_{int(time.time())}_{source_name}'
                    saved      = default_storage.save(tmp_path, ContentFile(img_bytes))
                    query_logo = default_storage.url(saved)

                    cache_key = f'logo_search_{hashlib.md5(img_bytes).hexdigest()}_{nama_merek_qry.lower()}'

                    # Cek cache DULUAN, sebelum OCR/embedding sama sekali --
                    # kalau gambar+nama ini persis udah pernah dicari (dalam 5 menit
                    # terakhir), langsung pakai hasil lama, skip semua komputasi berat.
                    cached_results = cache.get(cache_key)
                    if cached_results is not None:
                        results      = cached_results
                        hasil_rescue = [r for r in results if r['nama_cocok']]
                        hasil_visual = [r for r in results if r['lolos_vector'] and not r['nama_cocok']]
                    else:
                        validasi_ocr = request.POST.get('validasi_ocr') == '1'
                        teks_ocr = ''

                        if validasi_ocr:
                            t0 = time.time()
                            teks_ocr = extract_text_from_logo(Image.open(io.BytesIO(img_bytes)))
                            print(f"[TIMING] OCR: {time.time() - t0:.2f}s")

                            abaikan_peringatan = request.POST.get('abaikan_peringatan') == '1'
                            cocok = cek_kecocokan_nama_dan_ocr(nama_merek_qry, teks_ocr)

                            if nama_merek_qry and not abaikan_peringatan and not cocok:
                                context = {
                                    'peringatan_nama': True,
                                    'teks_terdeteksi': teks_ocr,
                                    'nama_merek_qry':  nama_merek_qry,
                                    'query_logo':      query_logo,
                                    'tmp_logo_path':   saved,
                                    'total_logo_db':   total_logo_db,
                                }
                                return render(request, 'logo_search.html', context)

                        t1 = time.time()
                        query_emb, info = extract_embedding_and_type(img_bytes)
                        print(f"[TIMING] extract_embedding_and_type: {time.time() - t1:.2f}s (jenis={info['jenis']}, rasio_teks={info['rasio_teks']:.2f})")

                        nama_untuk_cari = nama_merek_qry
                        if not nama_untuk_cari and info['teks']:
                            nama_untuk_cari = info['teks']
                            print(f'[INFO] nama merek kosong, pakai hasil OCR: "{nama_untuk_cari}"')

                        t2 = time.time()
                        results = cari_logo_mirip(query_emb, nama_untuk_cari, cache_key=cache_key, jenis=info['jenis'])
                        print(f"[TIMING] cari_logo_mirip: {time.time() - t2:.2f}s")

                        hasil_rescue = [r for r in results if r['nama_cocok']]
                        hasil_visual = [r for r in results if r['lolos_vector'] and not r['nama_cocok']]
                except Exception:
                    import traceback
                    traceback.print_exc()
                    error = 'Gagal memproses gambar. Coba lagi atau gunakan format lain.'

    context = {
        'hasil_visual':        hasil_visual,
        'hasil_rescue':        hasil_rescue,
        'query_logo':          query_logo,
        'tmp_logo_path':       saved if 'saved' in dir() else tmp_logo_path,
        'total_logo_db':       total_logo_db,
        'kemiripan_ditemukan': len(results),
        'error':               error,
        'nama_merek_qry':      nama_merek_qry,
    }

    return render(request, 'logo_search.html', context)

def merek_autocomplete_view(request):
    """Endpoint live-search buat field nama merek di logo_search.html."""
    query = request.GET.get('q', '')
    return JsonResponse({'results': cari_merek_autocomplete(query)})


@login_required
@user_passes_test(is_staff)
def logo_upload_view(request):
    if request.method == 'POST':
        nama_merek = request.POST.get('nama_merek', '').strip()
        logo_file  = request.FILES.get('logo')
        logo_url   = request.POST.get('logo_url', '').strip()

        if not nama_merek or (not logo_file and not logo_url):
            return JsonResponse(
                {'status': 'error', 'message': 'Nama merek dan file/URL logo wajib diisi.'},
                status=400
            )

        nama_merek_upper = nama_merek.upper()

        if logo_file:
            img_bytes   = logo_file.read()
            source_name = logo_file.name
        else:
            img_bytes, source_name, fetch_err = _fetch_image_from_url(logo_url)
            if fetch_err:
                return JsonResponse({'status': 'error', 'message': fetch_err}, status=400)

        valid, err_msg = validate_image_file(img_bytes, source_name)
        if not valid:
            return JsonResponse({'status': 'error', 'message': err_msg}, status=400)

        ext = source_name.rsplit('.', 1)[-1].lower()

        try:
            emb = extract_embedding(img_bytes)

            kemungkinan_duplikat = cari_logo_mirip(emb)
            duplikat_ketat = [d for d in kemungkinan_duplikat if d['similarity'] >= DUPLICATE_LOGO_THRESHOLD * 100]

            if duplikat_ketat:
                daftar_mirip = ', '.join(f"{d['nama_merek']} ({d['similarity']}%)" for d in duplikat_ketat)
                return JsonResponse({
                    'status': 'error',
                    'message': f'Logo ini sangat mirip dengan logo yang sudah terdaftar: {daftar_mirip}. '
                                f'Upload ditolak untuk mencegah duplikasi data.'
                }, status=400)

            entry             = LogoDatabaseEntry()
            entry.nama_merek  = nama_merek_upper
            entry.embedding   = embedding_to_str(emb)
            entry.uploaded_by = request.user
            webp_bytes, webp_filename = optimize_logo_image(img_bytes, source_name)
            entry.logo_file.save(webp_filename, ContentFile(webp_bytes), save=True)

            _models._disable_signals = True
            try:
                merek_obj, merek_created = DataMerek.objects.get_or_create(
                nama_merek=nama_merek_upper,
                defaults={'uploaded_by': request.user}
            )
            finally:
                _models._disable_signals = False

            notes = f'Upload logo: {nama_merek_upper}'
            if merek_created:
                notes += ' (merek baru otomatis ditambahkan)'
            if logo_url:
                notes += ' (via URL)'

            DatabaseChangeLog.objects.create(
                admin_user  = request.user,
                filename    = source_name,
                fileformat  = ext,
                rows_before = LogoDatabaseEntry.objects.count() - 1,
                rows_after  = LogoDatabaseEntry.objects.count(),
                status      = 'success',
                notes       = notes,
            )

            cache.clear()

            return JsonResponse({
                'status':  'ok',
                'message': f'Logo "{nama_merek_upper}" berhasil ditambahkan.',
                'id':      entry.id,
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Gagal: {str(e)}'}, status=500)

    logos = LogoDatabaseEntry.objects.all()
    return render(request, 'logo_upload.html', {'logos': logos})


@login_required
@user_passes_test(is_staff)
@require_http_methods(["POST"])
def logo_delete_view(request, logo_id):
    try:
        entry = LogoDatabaseEntry.objects.get(id=logo_id)
        nama  = entry.nama_merek
        if entry.logo_file:
            default_storage.delete(entry.logo_file.name)
        entry.delete()

        DatabaseChangeLog.objects.create(
            admin_user  = request.user,
            filename    = 'Manual Delete',
            fileformat  = 'manual',
            rows_before = LogoDatabaseEntry.objects.count() + 1,
            rows_after  = LogoDatabaseEntry.objects.count(),
            status      = 'deleted',
            notes       = f'Hapus logo: {nama}',
        )

        cache.clear()

        now = timezone.localtime()
        return JsonResponse({
            'status': 'ok', 'message': f'Logo "{nama}" dihapus.',
            'log_time': now.strftime('%d %b %Y'), 'log_time_detail': now.strftime('%H:%M:%S'),
            'log_admin': request.user.username, 'log_filename': 'Manual Delete',
            'log_notes': f'Hapus logo: {nama}', 'log_status': 'deleted',
        })
    except LogoDatabaseEntry.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Logo tidak ditemukan.'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@login_required
@user_passes_test(is_staff)
@require_http_methods(["POST"])
def logo_edit_view(request, logo_id):
    try:
        entry = LogoDatabaseEntry.objects.get(id=logo_id)
        nama_lama = entry.nama_merek
        nama_baru = request.POST.get('nama_merek', '').strip().upper()
        new_file = request.FILES.get('logo')
        logo_url = request.POST.get('logo_url', '').strip()

        if not nama_baru:
            return _tolak_edit_logo(request, nama_lama, 'Nama merek tidak boleh kosong.')

        ada_perubahan = False
        img_bytes = None
        new_filename = None
        old_file_name = None

        if new_file:
            img_bytes = new_file.read()
            new_filename = new_file.name
        elif logo_url:
            img_bytes, new_filename, fetch_err = _fetch_image_from_url(logo_url)
            if fetch_err:
                return _tolak_edit_logo(request, nama_lama, fetch_err)

        if img_bytes:
            valid, err_msg = validate_image_file(img_bytes, new_filename)
            if not valid:
                return _tolak_edit_logo(request, nama_lama, err_msg)

            emb = extract_embedding(img_bytes)
            kemungkinan_duplikat = cari_logo_mirip(emb)
            duplikat_ketat = [
                d for d in kemungkinan_duplikat
                if d['similarity'] >= DUPLICATE_LOGO_THRESHOLD * 100 and d.get('id') != entry.id
            ]

            if duplikat_ketat:
                daftar_mirip = ', '.join(f"{d['nama_merek']} ({d['similarity']}%)" for d in duplikat_ketat)
                return _tolak_edit_logo(
                    request, nama_lama,
                    f'Gambar ini sangat mirip dengan logo merek lain: {daftar_mirip}. Perubahan ditolak.'
                )

            webp_bytes, webp_filename = optimize_logo_image(img_bytes, new_filename)
            old_file_name = entry.logo_file.name if entry.logo_file else None
            entry.logo_file.save(webp_filename, ContentFile(webp_bytes), save=False)

            entry.embedding = embedding_to_str(emb)
            ada_perubahan = True

        if nama_baru != nama_lama:
            entry.nama_merek = nama_baru
            ada_perubahan = True
            _models._disable_signals = True
            try:
                DataMerek.objects.filter(nama_merek=nama_lama).update(nama_merek=nama_baru)
                DataMerek.objects.get_or_create(nama_merek=nama_baru)
            finally:
                _models._disable_signals = False

        if not ada_perubahan:
            return JsonResponse({'status': 'warning', 'message': 'Tidak ada perubahan.'}, status=200)

        entry.save()

        # Hapus file lama hanya setelah entri tersimpan, supaya gak ada entri yang menunjuk file hilang.
        if old_file_name and old_file_name != entry.logo_file.name:
            default_storage.delete(old_file_name)

        nama_berubah = nama_baru != nama_lama
        logo_berubah = bool(img_bytes)
        if nama_berubah and logo_berubah:
            pesan = f'Update nama merek dan logo sukses: {nama_lama} -> {nama_baru}'
        elif nama_berubah:
            pesan = f'Update nama merek sukses: {nama_lama} -> {nama_baru}'
        else:
            pesan = f'Update logo sukses: {nama_lama}'

        notes = pesan
        if logo_berubah:
            notes += ' (gambar diperbarui via url)' if logo_url else ' (gambar diperbarui via upload)'

        log = _catat_log_logo(request, 'success', notes)

        cache.clear()

        return JsonResponse({
            'status': 'ok', 'message': pesan, 'nama_baru': nama_baru,
            'diubah': {'nama_merek': nama_berubah, 'logo': logo_berubah},
            'logo_url': entry.logo_file.url if entry.logo_file else None,
            **log,
        })
    except LogoDatabaseEntry.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Logo tidak ditemukan.'}, status=404)
    except Exception as e:
        try:
            log = _catat_log_logo(request, 'failed', f'Gagal edit logo (id {logo_id}): {e}')
        except Exception:
            log = {}
        return JsonResponse({'status': 'error', 'message': str(e), **log}, status=500)