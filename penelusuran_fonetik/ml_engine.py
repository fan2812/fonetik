# ================================================================
# FILE BARU: penelusuran_fonetik/ml_engine.py
#
# Modul ML untuk ekstraksi fitur logo menggunakan ResNet50 pretrained.
# Tidak ada training dari nol — hanya feature extraction.
# Model yang dipakai sudah ditraining di 1.2 juta gambar ImageNet.
#
# Install:
#   pip install torch torchvision Pillow numpy easyocr
# ================================================================

import io
import json
import easyocr
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import time
from torch import nn
from rembg import remove, new_session
from .constants import LOGO_TEXT_RATIO_TEKS, LOGO_TEXT_RATIO_SIMBOL, LOGO_OCR_MIN_CONF

# Singleton — model hanya di-load sekali saat server start
_model     = None
_transform = None
_ocr_reader = None

def _load_model():
    """
    Load ResNet50 pretrained sekali, simpan sebagai singleton.
    Layer FC terakhir dihapus -> output 2048-dim feature vector.
    """
    global _model, _transform
    if _model is not None:
        return _model, _transform

    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    feature_extractor.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    _model     = feature_extractor
    _transform = transform
    return _model, _transform

_bg_removal_session = None

def _get_bg_removal_session():
    """Singleton -- model rembg cuma di-load sekali."""
    global _bg_removal_session
    if _bg_removal_session is None:
        _bg_removal_session = new_session('isnet-general-use')  # Menggunakan model IS-Net (pemenang eksperimen v4)
    return _bg_removal_session


def _auto_crop_to_content(img: Image.Image, padding_px=10, rembg_max_dim=800) -> Image.Image:
    """
    Buang background logo menggunakan model AI (rembg), lalu crop
    ke bounding box konten yang tersisa.

    PENTING: `img` di sini HARUS gambar asli (RGBA kalau file sumbernya
    punya alpha channel) — JANGAN di-convert('RGB') sebelum masuk sini.
    """
    # Downscale dulu sebelum masuk rembg -- rembg makin lambat sebanding
    # sama ukuran piksel, padahal hasil akhirnya di-resize ke 224x224 juga.
    # Logo biasanya gak butuh detail di atas ~800px buat background removal.
    work_img = img.copy()
    if max(work_img.size) > rembg_max_dim:
        work_img.thumbnail((rembg_max_dim, rembg_max_dim), Image.LANCZOS)

    session = _get_bg_removal_session()
    rgba = remove(work_img, session=session)
    arr = np.array(rgba)

    mask = arr[:, :, 3] > 30
    if not mask.any():
        return img.convert('RGB')

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - padding_px)
    cmin = max(0, cmin - padding_px)
    rmax = min(arr.shape[0], rmax + padding_px)
    cmax = min(arr.shape[1], cmax + padding_px)

    canvas = Image.new('RGB', rgba.size, (255, 255, 255))
    canvas.paste(rgba, (0, 0), rgba)
    return canvas.crop((cmin, rmin, cmax, rmax))

def _to_rgb_on_white(img, max_dim=1200):
    """Salinan RGB untuk OCR/jenis: tanpa rembg, transparansi ditempel ke putih."""
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        out = Image.new('RGB', rgba.size, (255, 255, 255))
        out.paste(rgba, (0, 0), rgba)
    else:
        out = img.convert('RGB')
    out.thumbnail((max_dim, max_dim), Image.LANCZOS)   # jaga OCR tetap cepat untuk upload besar
    return out

def _extract(image_source, with_text=False):
    model, transform = _load_model()

    # Buka gambar APA ADANYA dulu (pertahankan alpha channel kalau ada) —
    # jangan langsung .convert('RGB') di sini, karena _auto_crop_to_content
    # butuh channel alpha asli buat rembg. Convert ke RGB dilakukan
    # belakangan, setelah crop selesai.
    if isinstance(image_source, Image.Image):
        img = image_source
    elif isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, str):
        img = Image.open(image_source)
    else:
        img = Image.open(image_source)

    # Salinan untuk OCR/jenis: SEBELUM rembg, karena rembg bisa menghapus bagian huruf
    img_ocr = _to_rgb_on_white(img) if with_text else None

    # 1. Ukur waktu proses rembg / auto crop
    t0 = time.time()
    img = _auto_crop_to_content(img)       # buang whitespace/margin dulu (rgba asli masuk sini)
    print(f"[TIMING] crop: {time.time() - t0:.2f}s")
    # img.save(f'C:/fonetik/debug_crop_{int(time.time())}.png')   # debug: aktifkan hanya saat perlu
    info = None
    if with_text:
        t_ocr = time.time()
        teks, rasio, jenis = analyze_logo_text(img_ocr)   # gambar asli (bukan hasil rembg)
        info = {'teks': teks, 'rasio_teks': rasio, 'jenis': jenis}
        print(f"[TIMING] ocr: {time.time() - t_ocr:.2f}s")

    # 2. Ukur waktu proses resize, transform, dan ekstraksi fitur ResNet
    t1 = time.time()
    img = _resize_with_padding(img, 224)   # baru resize setelah di-crop
    #img = img.convert('L').convert('RGB')  # grayscale
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        emb = model(tensor).squeeze()
    print(f"[TIMING] resnet: {time.time() - t1:.2f}s")

    # Normalisasi embedding vector
    norm = emb.norm()
    if norm > 0:
        emb = emb / norm

    emb_list = emb.numpy().tolist()
    return (emb_list, info) if with_text else emb_list

def extract_embedding(image_source) -> list:
    return _extract(image_source, with_text=False)


def extract_embedding_and_type(image_source):
    """Return (embedding, {'teks','rasio_teks','jenis'})."""
    return _extract(image_source, with_text=True)

def _resize_with_padding(img, target_size=224):
    """
    Resize gambar mempertahankan aspect ratio, lalu padding ke persegi
    dengan background putih.
    """
    img.thumbnail((target_size, target_size), Image.LANCZOS)
    canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    offset_x = (target_size - img.width) // 2
    offset_y = (target_size - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))
    return canvas

def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Cosine similarity antara dua embedding yang sudah L2-normalized.
    Returns float 0.0–1.0.
    """
    a   = np.array(vec1, dtype=np.float32)
    b   = np.array(vec2, dtype=np.float32)
    dot = float(np.dot(a, b))
    return max(0.0, min(1.0, dot))

def _get_ocr_reader():
    """Singleton -- model EasyOCR cuma di-load sekali. `return` HARUS di luar
    blok if: kalau di dalam, panggilan kedua dst (reader udah ke-cache) bakal
    nembus ke akhir function tanpa return apapun alias balikin None diam-diam."""
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader

def extract_text_from_logo(img: Image.Image, max_dim=1000) -> str:
    """Baca teks yang ada di gambar logo. Return string kosong kalau gak ada teks kebaca."""
    reader = _get_ocr_reader()
    
    if max(img.size) > max_dim:
        img = img.copy()
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        
    arr = np.array(img.convert('RGB'))
    hasil = reader.readtext(arr, detail=0)
    return ' '.join(hasil).strip()

def analyze_logo_text(img: Image.Image, max_dim=1000):
    #Input: gambar logo yang SUDAH bersih (hasil _auto_crop_to_content, background putih).
    #Return (teks, rasio_teks, jenis). Logikanya sama dengan classify_logos_v5.py.
    reader = _get_ocr_reader()
    img = img.convert('RGB')
    if max(img.size) > max_dim:
        img = img.copy()
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    arr = np.array(img)
    content = np.array(img.convert('L')) < 235
    teks_mask = np.zeros(content.shape, dtype=bool)
    kata = []
    for bbox, teks, conf in reader.readtext(arr, detail = 1):
        if conf < LOGO_OCR_MIN_CONF or not teks.strip():
            continue 
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x0, x1 = max(0, int(min(xs))), min(arr.shape[1], int(max(xs)))
        y0, y1 = max(0, int(min(ys))), min(arr.shape[0], int(max(ys)))
        teks_mask[y0:y1, x0:x1] = True
        kata.append(teks.strip())
    total = int (content.sum())
    rasio = float((content & teks_mask).sum()) / total if total else 0.0
    if rasio >= LOGO_TEXT_RATIO_TEKS:
        jenis = "teks"
    elif rasio <= LOGO_TEXT_RATIO_SIMBOL:
        jenis = "simbol"
    else:
        jenis = "campuran"
    return ' '.join(kata), rasio, jenis

def embedding_to_str(embedding: list) -> str:
    return json.dumps(embedding, separators=(',', ':'))


def str_to_embedding(s: str) -> list:
    return json.loads(s) if s else []