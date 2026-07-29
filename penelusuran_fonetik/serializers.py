from rest_framework import serializers

from .models import DataMerek


class DataMerekSerializer(serializers.ModelSerializer):
    """Representasi dasar satu record DataMerek (id, nama, tanggal input)."""

    class Meta:
        model = DataMerek
        fields = ['id', 'nama_merek', 'created_at']


class PhoneticResultSerializer(serializers.Serializer):
    """
    BUKAN ModelSerializer -- cari_fonetik() di services.py mengembalikan
    list of dict, bukan queryset/model instance. Nama field di bawah ini
    WAJIB sama persis (termasuk ejaan) dengan key dict hasil cari_fonetik():
        {'merek_input': ..., 'merek_db': ..., 'similarity': ...}
    """
    merek_input = serializers.CharField()
    merek_db = serializers.CharField()
    similarity = serializers.FloatField()


class LogoSimilarResultSerializer(serializers.Serializer):
    """
    Field di bawah ini menyesuaikan key dict hasil cari_logo_mirip():
        {'nama_merek': ..., 'logo_url': ..., 'similarity': ...}
    logo_url pakai CharField (bukan URLField) karena get_logo_url()
    mengembalikan path relatif (/media/logos/xxx.jpg), bukan URL absolut.
    """
    nama_merek = serializers.CharField()
    logo_url = serializers.CharField(allow_null=True, required=False)
    similarity = serializers.FloatField()


class PhoneticSearchRequestSerializer(serializers.Serializer):
    """Validasi input body request untuk endpoint search fonetik."""
    query = serializers.CharField(max_length=255, allow_blank=False, required=True)