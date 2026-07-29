import io
import time
import pandas as pd
import openpyxl

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import logout, login
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.db import transaction
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.views.decorators.http import require_http_methods

from .models import DatabaseChangeLog, DataMerek, LogoDatabaseEntry
from .forms import UploadCSVForm
from .ml_engine import extract_embedding, embedding_to_str
from .services import cari_fonetik, validate_image_file, cari_logo_mirip, cleanup_tmp_files
from .constants import ALLOWED_CSV_EXT, MAX_CSV_SIZE_MB, CSV_ENCODINGS_TRY

MANUAL_MIN_ROWS = 21   # di bawah ini wajib manual, bukan file

def is_staff(user):
    return user.is_authenticated and user.is_staff

def is_admin(user):
    return user.is_authenticated and user.is_staff


# ================================================================
# 1. PENCARIAN FONETIK (publik)
# ================================================================

def search_view(request):
    query   = request.GET.get('q', '').strip()
    results = cari_fonetik(query) if query else []

    context = {
        'query':               query,
        'results':             results,
        'total_database':      DataMerek.objects.count(),
        'kemiripan_ditemukan': len(results),
    }
    return render(request, 'penelusuran_fonetik/result.html', context)


# ================================================================
# 2. UPDATE DB MEREK (admin)  — sebelumnya: upload_database
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
        obj, created = DataMerek.objects.get_or_create(nama_merek=nama_merek)
        rows_after   = DataMerek.objects.count()

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

    # ── B. Upload file (Excel/CSV) — minimal MANUAL_MIN_ROWS baris ─
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

                # Validasi minimum baris
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
                    for merek_nama in clean_values:
                        DataMerek.objects.get_or_create(nama_merek=merek_nama)

            else:  # Excel
                wb      = openpyxl.load_workbook(uploaded_file, read_only=True)
                ws      = wb.active

                headers = [
                    str(cell.value).strip().lower() if cell.value else ''
                    for cell in ws[1]
                ]
                merek_col_idx = 0
                for i, h in enumerate(headers):
                    if 'merek' in h:
                        merek_col_idx = i
                        break

                # Hitung baris data (exclude header)
                data_rows = list(ws.iter_rows(min_row=2, values_only=True))
                data_rows = [r for r in data_rows if r[merek_col_idx] if len(r) > merek_col_idx]

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
                    for row in data_rows:
                        val = row[merek_col_idx]
                        if val:
                            nama = str(val).strip().upper()
                            if nama:
                                DataMerek.objects.get_or_create(nama_merek=nama)
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
    # 'from' query param: 'fonetik' (default) atau 'logo'
    from_page    = request.GET.get('from', 'fonetik')
    
    merek_query = request.GET.get('mq', '').strip()
    merek_list = None
    if from_page == 'fonetik':
        qs = DataMerek.objects.all().order_by('nama_merek')
        if merek_query:
            qs = qs.filter(nama_merek__icontains=merek_query)
            merek_list = qs[:100]
    context = {
        'logs':         logs,
        'total_brands': total_brands,
        'total_logos':  total_logos,
        'total_logs':   total_logs,
        'from_page':    from_page,
        'merek_query':  merek_query,
        'merek_list': merek_list
    }
    return render(request, 'log.html', context)

@login_required
@user_passes_test(is_staff)
@require_http_methods(["POST"])
def merek_delete_view(request, merek_id):
    try :
        merek = DataMerek.objects.get(id=merek_id)
        nama = merek.nama_merek
        rows_before = DataMerek.objects.count()
        merek.delete()
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
        return JsonResponse({'status': 'ok', 'message': f'Merek "{nama}" dihapus.'})
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

        merek.nama_merek = nama_baru
        merek.save()

        DatabaseChangeLog.objects.create(
            admin_user  = request.user,
            filename    = 'Manual Edit',
            fileformat  = 'manual',
            rows_before = DataMerek.objects.count(),
            rows_after  = DataMerek.objects.count(),
            status      = 'success',
            notes       = f'Edit manual: {nama_lama} -> {nama_baru}',
        )

        return JsonResponse({'status': 'ok', 'message': f'Berhasil diubah menjadi "{nama_baru}".', 'nama_baru': nama_baru})
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

    results       = []
    query_logo    = None
    error         = None
    total_logo_db = LogoDatabaseEntry.objects.count()

    if request.method == 'POST':
        uploaded = request.FILES.get('logo')

        if not uploaded:
            error = 'Tidak ada file yang dipilih.'
        else:
            img_bytes = uploaded.read()
            valid, err_msg = validate_image_file(img_bytes, uploaded.name)

            if not valid:
                error = err_msg
            else:
                try:
                    tmp_path   = f'tmp/query_{int(time.time())}_{uploaded.name}'
                    saved      = default_storage.save(tmp_path, ContentFile(img_bytes))
                    query_logo = default_storage.url(saved)

                    query_emb = extract_embedding(img_bytes)
                    results   = cari_logo_mirip(query_emb)

                except Exception:
                    error = 'Gagal memproses gambar. Coba lagi atau gunakan format lain.'

    context = {
        'results':             results,
        'query_logo':          query_logo,
        'total_logo_db':       total_logo_db,
        'kemiripan_ditemukan': len(results),
        'error':               error,
    }
    return render(request, 'logo_search.html', context)


@login_required
@user_passes_test(is_staff)
def logo_upload_view(request):
    if request.method == 'POST':
        nama_merek = request.POST.get('nama_merek', '').strip()
        logo_file  = request.FILES.get('logo')

        if not nama_merek or not logo_file:
            return JsonResponse(
                {'status': 'error', 'message': 'Nama merek dan file logo wajib diisi.'},
                status=400
            )

        img_bytes = logo_file.read()

        valid, err_msg = validate_image_file(img_bytes, logo_file.name)
        if not valid:
            return JsonResponse({'status': 'error', 'message': err_msg}, status=400)

        ext = logo_file.name.rsplit('.', 1)[-1].lower()

        try:
            emb = extract_embedding(img_bytes)

            entry            = LogoDatabaseEntry()
            entry.nama_merek = nama_merek.upper()
            entry.embedding  = embedding_to_str(emb)
            entry.logo_file.save(logo_file.name, ContentFile(img_bytes), save=True)

            DatabaseChangeLog.objects.create(
                admin_user  = request.user,
                filename    = logo_file.name,
                fileformat  = ext,
                rows_before = LogoDatabaseEntry.objects.count() - 1,
                rows_after  = LogoDatabaseEntry.objects.count(),
                status      = 'success',
                notes       = f'Upload logo: {nama_merek.upper()}',
            )

            return JsonResponse({
                'status':  'ok',
                'message': f'Logo "{nama_merek.upper()}" berhasil ditambahkan.',
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
        return JsonResponse({'status': 'ok', 'message': f'Logo "{nama}" dihapus.'})
    except LogoDatabaseEntry.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Logo tidak ditemukan.'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)