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

def _auto_crop_to_content(img: Image.Image, bg_threshold=245, padding_px=10) -> Image.Image:
    rgb = img.convert('RGB')
    arr = np.array(rgb.convert('L'))
    
    # 1. Perbaikan Typo: bg_threshold
    mask = arr < bg_threshold
    
    if not mask.any():
        return rgb 
    
    # 2. Perbaikan Axis: Pisahkan rows (axis=1) dan cols (axis=0)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # 3. Ambil indeks bounding box
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # 4. Perbaikan Padding & Boundary Clamping (Height vs Width)
    rmin = max(0, rmin - padding_px)
    rmax = min(arr.shape[0], rmax + padding_px)
    cmin = max(0, cmin - padding_px)
    cmax = min(arr.shape[1], cmax + padding_px)
    
    return rgb.crop((cmin, rmin, cmax, rmax))

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


def extract_embedding(image_source) -> list:
    """
    Ekstraksi embedding 2048-dim dari gambar logo menggunakan ResNet50
    dengan preprocessing Grayscale + Padding 224x224.
    """
    model, transform = _load_model()

    if isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source)).convert('RGB')
    elif isinstance(image_source, str):
        img = Image.open(image_source).convert('RGB')
    else:
        img = Image.open(image_source).convert('RGB')
        
    img = _auto_crop_to_content(img)
    # Preprocessing pipeline berdasarkan hasil eksperimen v2
    img = _resize_with_padding(img, 224)
    img = img.convert('L').convert('RGB')  # ★ Grayscale preprocessing (Accuracy 0.955 vs Baseline 0.894)
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        emb = model(tensor).squeeze()

    norm = emb.norm()
    if norm > 0:
        emb = emb / norm

    return emb.numpy().tolist()


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