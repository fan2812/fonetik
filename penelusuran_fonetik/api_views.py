from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .ml_engine import extract_embedding          # ← sudah ada di ml_engine.py
from .services import cari_fonetik, cari_logo_mirip, validate_image_file
from .serializers import (
    PhoneticSearchRequestSerializer,
    PhoneticResultSerializer,
    LogoSimilarResultSerializer,
)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_search_fonetik(request):
    """
    POST /api/v1/search/fonetik/
    Body JSON: { "query": "nama merek" }
    """
    req_ser = PhoneticSearchRequestSerializer(data=request.data)
    req_ser.is_valid(raise_exception=True)

    query = req_ser.validated_data['query']
    hasil = cari_fonetik(query)

    return Response({
        'query':   query,
        'count':   len(hasil),
        'results': PhoneticResultSerializer(hasil, many=True).data,
    }, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_search_logo(request):
    """
    POST /api/v1/search/logo/
    Body multipart/form-data: logo = <file gambar>
    """
    uploaded = request.FILES.get('logo')
    if not uploaded:
        return Response(
            {'detail': 'File logo wajib diupload (field "logo").'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    img_bytes = uploaded.read()
    is_valid, err_msg = validate_image_file(img_bytes, uploaded.name)
    if not is_valid:
        return Response({'detail': err_msg}, status=status.HTTP_400_BAD_REQUEST)

    # extract_embedding() di ml_engine.py langsung terima raw bytes
    query_emb = extract_embedding(img_bytes)
    hasil     = cari_logo_mirip(query_emb)

    return Response({
        'count':   len(hasil),
        'results': LogoSimilarResultSerializer(hasil, many=True).data,
    }, status=status.HTTP_200_OK)