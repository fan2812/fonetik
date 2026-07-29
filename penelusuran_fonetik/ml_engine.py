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
    Layer FC terakhir dihapus → output 2048-dim feature vector.
    Download otomatis ~100MB pertama kali dijalankan.
    """
    global _model, _transform
    if _model is not None:
        return _model, _transform

    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Buang classifier layer terakhir → dapat 2048-dim embedding
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    feature_extractor.eval()

    # Preprocessing standar ImageNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    _model     = feature_extractor
    _transform = transform
    return _model, _transform


def extract_embedding(image_source) -> list:
    """
    Ekstrak feature vector 2048-dim dari gambar, lalu L2-normalize.

    Args:
        image_source: path string, raw bytes, atau file-like object

    Returns:
        List of 2048 floats (L2-normalized, siap untuk cosine sim)
    """
    import torch

    model, transform = _load_model()

    if isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source)).convert('RGB')
    elif isinstance(image_source, str):
        img = Image.open(image_source).convert('RGB')
    else:
        img = Image.open(image_source).convert('RGB')

    tensor = transform(img).unsqueeze(0)   # [1, 3, 224, 224]

    with torch.no_grad():
        emb = model(tensor).squeeze()       # [2048]

    # L2 normalize → cosine sim == dot product (lebih cepat)
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