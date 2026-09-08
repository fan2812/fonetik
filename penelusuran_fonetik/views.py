import hashlib
import io
import time
import requests
import pandas as pd
import openpyxl
from urllib.parse import urlparse

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

from .models import DatabaseChangeLog, DataMerek, LogoDatabaseEntry
from .forms import UploadCSVForm
from .ml_engine import extract_embedding, embedding_to_str
from .services import (
    cari_fonetik, validate_image_file, cari_logo_mirip, cleanup_tmp_files,
    cari_merek_autocomplete,
)
from .constants import ALLOWED_CSV_EXT, DUPLICATE_LOGO_THRESHOLD, MAX_CSV_SIZE_MB, CSV_ENCODINGS_TRY

MANUAL_MIN_ROWS = 21   # Di bawah ini wajib manual, bukan file
MAX_URL_IMAGE_MB = 5


def is_staff(user):
    return user.is_authenticated and user.is_staff

def is_admin(user):
    return user.is_authenticated and user.is_staff


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def _fetch_image_from_url(url):
    """Ambil gambar dari URL, validasi ukuran & content-type. Return (img_bytes, filename, error)."""
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return None, None, 'URL tidak valid.'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    try:
        resp = requests.get(url, timeout=10, stream=True, headers=headers)
        resp.raise_for_status()
    except requests.RequestException:
        return None, None, 'Gagal mengambil gambar dari URL tersebut.'

    content_type = resp.headers.get('Content-Type', '')
    if not content_type.startswith('image/'):
        return None, None, 'URL tidak mengarah ke file gambar.'

    content_length = resp.headers.get('Content-Length')
    if content_length and int(content_length) > MAX_URL_IMAGE_MB * 1024 * 1024:
        return None, None, f'Gambar terlalu besar. Maks {MAX_URL_IMAGE_MB}MB.'

    img_bytes = resp.content
    if len(img_bytes) > MAX_URL_IMAGE_MB * 1024 * 1024:
        return None, None, f'Gambar terlalu besar. Maks {MAX_URL_IMAGE_MB}MB.'

    filename = parsed.path.rsplit('/', 1)[-1] or 'logo_from_url.jpg'
    if '.' not in filename:
        filename += '.jpg'

    return img_bytes, filename, None


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
    from_page    = request.GET.get('from', 'fonetik')

    merek_query      = request.GET.get('mq', '').strip()
    merek_list       = None
    logos_list       = None
    merek_tanpa_logo = None

    if from_page == 'fonetik':
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
            status      = 'success',
            notes       = f'Hapus manual: {nama}',
        )

        cache.clear()

        now = timezone.localtime()
        return JsonResponse({
            'status': 'ok', 'message': f'Merek "{nama}" dihapus.',
            'log_time': now.strftime('%d %b %Y'), 'log_time_detail': now.strftime('%H:%M:%S'),
            'log_admin': request.user.username, 'log_filename': 'Manual Delete',
            'log_notes': f'Hapus manual: {nama}', 'log_status': 'success',
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

    results        = []
    hasil_visual   = []
    hasil_rescue   = []
    query_logo     = None
    error          = None
    nama_merek_qry = ''
    total_logo_db  = LogoDatabaseEntry.objects.count()

    if request.method == 'POST':
        uploaded       = request.FILES.get('logo')
        logo_url       = request.POST.get('logo_url', '').strip()
        nama_merek_qry = request.POST.get('nama_merek', '').strip()

        if not uploaded and not logo_url:
            error = 'Tidak ada file atau URL yang diberikan.'
        else:
            if uploaded:
                img_bytes = uploaded.read()
                source_name = uploaded.name
            else:
                img_bytes, source_name, fetch_err = _fetch_image_from_url(logo_url)
                if fetch_err:
                    error = fetch_err

            if not error:
                valid, err_msg = validate_image_file(img_bytes, source_name)

                if not valid:
                    error = err_msg
                else:
                    try:
                        tmp_path   = f'tmp/query_{int(time.time())}_{source_name}'
                        saved      = default_storage.save(tmp_path, ContentFile(img_bytes))
                        query_logo = default_storage.url(saved)

                        cache_key   = f'logo_search_{hashlib.md5(img_bytes).hexdigest()}_{nama_merek_qry.lower()}'
                        query_emb   = extract_embedding(img_bytes)
                        
                        # Langsung panggil cari_logo_mirip dengan menyertakan nama_merek_qry dan cache_key
                        results = cari_logo_mirip(query_emb, nama_merek_qry, cache_key=cache_key)
                        hasil_visual = [r for r in results if not r.get('rescued_by_nama')]
                        hasil_rescue = [r for r in results if r.get('rescued_by_nama')]

                    except Exception:
                        error = 'Gagal memproses gambar. Coba lagi atau gunakan format lain.'

        context = {
        'hasil_visual':        hasil_visual,
        'hasil_rescue':        hasil_rescue,
        'query_logo':          query_logo,
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

        if LogoDatabaseEntry.objects.filter(nama_merek=nama_merek_upper).exists():
            return JsonResponse({
                'status': 'error',
                'message': f'Merek "{nama_merek_upper}" sudah memiliki logo terdaftar. '
                            f'Gunakan fitur Edit di halaman Log Database untuk mengganti logo yang sudah ada.'
            }, status=400)

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
            entry.logo_file.save(source_name, ContentFile(img_bytes), save=True)

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
            status      = 'success',
            notes       = f'Hapus logo: {nama}',
        )

        cache.clear()

        now = timezone.localtime()
        return JsonResponse({
            'status': 'ok', 'message': f'Logo "{nama}" dihapus.',
            'log_time': now.strftime('%d %b %Y'), 'log_time_detail': now.strftime('%H:%M:%S'),
            'log_admin': request.user.username, 'log_filename': 'Manual Delete',
            'log_notes': f'Hapus logo: {nama}', 'log_status': 'success',
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
            return JsonResponse({'status': 'error', 'message': 'Nama merek tidak boleh kosong.'}, status=400)

        ada_perubahan = False
        duplikat_ketat = []
        img_bytes = None 
        new_filename = None

        if new_file:
            img_bytes = new_file.read()
            new_filename = new_file.name
        elif logo_url:
            img_bytes, new_filename, fetch_err = _fetch_image_from_url(logo_url)
            if fetch_err:
                return JsonResponse({'status': 'error', 'message': fetch_err}, status=400)
            
        if img_bytes:
            valid, err_msg = validate_image_file(img_bytes, new_filename)
            if not valid:
                return JsonResponse({'status': 'error', 'message': err_msg}, status=400)

            emb = extract_embedding(img_bytes)
            kemungkinan_duplikat = cari_logo_mirip(emb)
            duplikat_ketat = [d for d in kemungkinan_duplikat if d['similarity'] >= DUPLICATE_LOGO_THRESHOLD * 100 and d['nama_merek'] != nama_lama]

            if duplikat_ketat:
                daftar_mirip = ', '.join(f"{d['nama_merek']} ({d['similarity']}%)" for d in duplikat_ketat)
                return JsonResponse({
                    'status': 'error',
                    'message': f'Gambar ini sangat mirip dengan logo merek lain: {daftar_mirip}. Perubahan ditolak.'
                }, status=400)

            if entry.logo_file:
                default_storage.delete(entry.logo_file.name)
            entry.logo_file.save(new_filename, ContentFile(img_bytes), save=False)

            entry.embedding = embedding_to_str(emb)
            ada_perubahan = True

        if nama_baru != nama_lama:
            entry.nama_merek = nama_baru
            ada_perubahan = True
            DataMerek.objects.filter(nama_merek=nama_lama).update(nama_merek=nama_baru)
            DataMerek.objects.get_or_create(nama_merek=nama_baru)

        if not ada_perubahan:
            return JsonResponse({'status': 'warning', 'message': 'Tidak ada perubahan.'}, status=200)

        entry.save()

        notes = f'Edit logo: {nama_lama} -> {nama_baru}' if nama_lama != nama_baru else f'Edit gambar logo: {nama_lama}'
        if img_bytes:
            notes += ' (gambar diperbarui via url)' if logo_url else ' (gambar diperbarui via upload)'

        DatabaseChangeLog.objects.create(
            admin_user  = request.user,
            filename    = 'Manual Edit',
            fileformat  = 'manual',
            rows_before = LogoDatabaseEntry.objects.count(),
            rows_after  = LogoDatabaseEntry.objects.count(),
            status      = 'success',
            notes       = notes,
        )

        cache.clear()

        now = timezone.localtime()
        return JsonResponse({
            'status': 'ok', 'message': f'Logo "{nama_lama}" berhasil diubah menjadi "{nama_baru}".', 'nama_baru': nama_baru,
            'logo_url': entry.logo_file.url if entry.logo_file else None,
            'log_time': now.strftime('%d %b %Y'), 'log_time_detail': now.strftime('%H:%M:%S'),
            'log_admin': request.user.username, 'log_filename': 'Manual Edit',
            'log_notes': notes, 'log_status': 'success',
        })
    except LogoDatabaseEntry.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Logo tidak ditemukan.'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)