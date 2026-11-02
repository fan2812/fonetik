import csv
from collections import Counter

from django.core.management.base import BaseCommand
from PIL import Image

from penelusuran_fonetik import constants as C
from penelusuran_fonetik.ml_engine import analyze_logo_text
from penelusuran_fonetik.models import LogoDatabaseEntry

TEKS_MIN = C.LOGO_TEXT_RATIO_TEKS
SIMBOL_MAX = C.LOGO_TEXT_RATIO_SIMBOL
MARGIN = getattr(C, "LOGO_TEXT_RATIO_MARGIN", 0.05)  # fallback kalau belum ditambah ke constants.py

REF = {"teks": 44, "campuran": 39, "simbol": 17}  # persen, dari eksperimen v5 (n=134)


def _file_field(model):
    for f in model._meta.get_fields():
        if f.get_internal_type() in ("FileField", "ImageField"):
            return f.name


def _klasifikasi_mentah(path):
    """OCR di gambar asli (tanpa rembg), persis seperti yang dipakai pencarian.
    Cepat: tanpa rembg dan ResNet. Hasilnya sama dengan jalur production."""
    from penelusuran_fonetik.ml_engine import _to_rgb_on_white

    with Image.open(path) as img:
        teks, rasio, jenis = analyze_logo_text(_to_rgb_on_white(img))
    return rasio, jenis


def _klasifikasi_production(path):
    """Lewat `_extract` penuh (rembg + ResNet ikut jalan, jauh lebih lambat).
    Dipakai hanya untuk verifikasi silang; klasifikasinya sama dengan mode default."""
    from penelusuran_fonetik.ml_engine import extract_embedding_and_type

    # === ADAPTER: sesuaikan argumen dengan signature extract_embedding_and_type ===
    _, info = extract_embedding_and_type(path)
    # ==============================================================================
    return info["rasio_teks"], info["jenis"]


class Command(BaseCommand):
    help = "Distribusi jenis logo (teks/campuran/simbol) di seluruh DB + zona abu-abu rasio_teks"

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=0, help="0 = semua")
        parser.add_argument(
            "--production", action="store_true",
            help="Pakai `_extract` penuh (rembg + ResNet ikut jalan, lambat; untuk verifikasi silang)",
        )
        parser.add_argument("--csv", type=str, default="rasio_teks.csv")

    def handle(self, *args, **opts):
        ff = _file_field(LogoDatabaseEntry)
        qs = LogoDatabaseEntry.objects.all().order_by("id")
        if opts["limit"]:
            qs = qs[: opts["limit"]]

        klasifikasi = _klasifikasi_production if opts["production"] else _klasifikasi_mentah
        self.stdout.write(
            f"Mode: {'production penuh (rembg + ResNet)' if opts['production'] else 'OCR gambar asli (= klasifikasi pencarian)'}"
        )
        self.stdout.write(f"Batas: teks >= {TEKS_MIN} | simbol <= {SIMBOL_MAX} | margin {MARGIN}\n")

        counter, failed, rows = Counter(), [], []
        total = opts["limit"] or qs.count()

        for n, e in enumerate(qs, 1):
            try:
                rasio, jenis = klasifikasi(getattr(e, ff).path)
                counter[jenis] += 1
                rows.append((e.id, rasio, jenis))
            except Exception as ex:
                failed.append((e.id, str(ex)[:60]))
            if n % 25 == 0:
                self.stdout.write(f"  progress {n}/{total}")

        # --- ringkasan distribusi ---
        ok = sum(counter.values())
        self.stdout.write(f"\n{'JENIS':<10} {'JUMLAH':>7} {'PRODUCTION':>11} {'EKSPERIMEN':>11}")
        self.stdout.write("-" * 44)
        for j in ("teks", "campuran", "simbol"):
            pct = counter[j] / ok * 100 if ok else 0
            self.stdout.write(f"{j:<10} {counter[j]:>7} {pct:>10.1f}% {REF[j]:>10}%")
        self.stdout.write(f"\nBerhasil: {ok} | Gagal: {len(failed)}")
        for i, err in failed[:10]:
            self.stdout.write(f"  gagal id={i}: {err}")

        # --- simpan CSV ---
        with open(opts["csv"], "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "rasio_teks", "jenis"])
            w.writerows(rows)
        self.stdout.write(f"\nDisimpan ke {opts['csv']}")

        # --- histogram rasio_teks ---
        vals = [r for _, r, _ in rows if r is not None]
        bins = [
            (0.00, 0.05), (0.05, 0.10), (0.10, SIMBOL_MAX), (SIMBOL_MAX, SIMBOL_MAX + MARGIN),
            (SIMBOL_MAX + MARGIN, 0.50), (0.50, TEKS_MIN - MARGIN),
            (TEKS_MIN - MARGIN, TEKS_MIN), (TEKS_MIN, 1.01),
        ]
        self.stdout.write("\nHistogram rasio_teks:")
        for lo, hi in bins:
            c = sum(1 for v in vals if lo <= v < hi)
            self.stdout.write(f"  {lo:.2f}-{min(hi, 1.0):.2f} | {c:>4} {'#' * (c // 8)}")

        # --- zona abu-abu ---
        dekat_teks = sum(1 for v in vals if TEKS_MIN - MARGIN <= v < TEKS_MIN)
        dekat_simbol = sum(1 for v in vals if SIMBOL_MAX < v <= SIMBOL_MAX + MARGIN)
        n_vals = len(vals) or 1
        self.stdout.write(
            f"\nZona abu-abu: dekat batas teks ({TEKS_MIN - MARGIN:.2f}-{TEKS_MIN:.2f}) = "
            f"{dekat_teks} ({dekat_teks / n_vals * 100:.1f}%) | "
            f"dekat batas simbol ({SIMBOL_MAX:.2f}-{SIMBOL_MAX + MARGIN:.2f}) = "
            f"{dekat_simbol} ({dekat_simbol / n_vals * 100:.1f}%)"
        )