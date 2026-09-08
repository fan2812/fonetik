# ================================================================
# FILE BARU: penelusuran_fonetik/ml_engine.py
#
# Modul ML untuk ekstraksi fitur logo menggunakan ResNet50 pretrained.
# Tidak ada training dari nol — hanya feature extraction.
# Model yang dipakai sudah ditraining di 1.2 juta gambar ImageNet.
#
# Install:
#   pip install torch torchvision Pillow numpy
# ================================================================

import io
import json
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
from rembg import remove, new_session

# Singleton — model hanya di-load sekali saat server start
_model     = None
_transform = None


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


def _auto_crop_to_content(img: Image.Image, padding_px=10) -> Image.Image:
    """
    Buang background logo menggunakan model AI (rembg), lalu crop
    ke bounding box konten yang tersisa. Menggantikan pendekatan heuristik
    berbasis sampling warna, yang gagal pada background bertekstur/pattern.
    """
    session = _get_bg_removal_session()
    rgba = remove(img.convert('RGB'), session=session)
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

def extract_embedding(image_source) -> list:
    model, transform = _load_model()

    if isinstance(image_source, Image.Image):
        img = image_source.convert('RGB')
    elif isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source)).convert('RGB')
    elif isinstance(image_source, str):
        img = Image.open(image_source).convert('RGB')
    else:
        img = Image.open(image_source).convert('RGB')

    img = _auto_crop_to_content(img)       # buang whitespace/margin dulu
    img = _resize_with_padding(img, 224)   # baru resize setelah di-crop
    img = img.convert('L').convert('RGB')  # grayscale (sudah ada dari sebelumnya)

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        emb = model(tensor).squeeze()

    norm = emb.norm()
    if norm > 0:
        emb = emb / norm

    return emb.numpy().tolist()

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


def embedding_to_str(embedding: list) -> str:
    return json.dumps(embedding, separators=(',', ':'))


def str_to_embedding(s: str) -> list:
    return json.loads(s) if s else []