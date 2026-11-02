import csv
import json
import numpy as np
from django.core.management.base import BaseCommand
from penelusuran_fonetik.models import LogoDatabaseEntry


def _find_field(model, *hints):
    for f in model._meta.get_fields():
        if any(h in f.name.lower() for h in hints):
            return f.name
    return None


def _to_vector(raw):
    """Handle embedding yang disimpan sebagai list/JSON/bytes."""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(raw), dtype=np.float32)
    if isinstance(raw, str):
        raw = json.loads(raw)
    return np.asarray(raw, dtype=np.float32)


class Command(BaseCommand):
    help = "Cari logo duplikat/nyaris identik berdasarkan cosine similarity embedding"

    def add_arguments(self, parser):
        parser.add_argument("--min-score", type=float, default=0.99)
        parser.add_argument("--csv", type=str, default="duplicates.csv")

    def handle(self, *args, **opts):
        emb_field = _find_field(LogoDatabaseEntry, "embed", "vector")
        if not emb_field:
            self.stderr.write("Field embedding gak ketemu, cek nama field di models.py")
            return
        self.stdout.write(f"Field embedding: {emb_field}")

        entries, vecs = [], []
        for e in LogoDatabaseEntry.objects.all().order_by("id"):
            v = _to_vector(getattr(e, emb_field))
            if v is not None and v.size:
                entries.append(e)
                vecs.append(v)

        M = np.vstack(vecs)
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        S = M @ M.T
        iu, ju = np.triu_indices(len(entries), k=1)
        mask = S[iu, ju] >= opts["min_score"]
        pairs = sorted(
            zip(iu[mask], ju[mask], S[iu, ju][mask]), key=lambda x: -x[2]
        )

        self.stdout.write(f"\nTotal entry: {len(entries)} | Pasangan >= {opts['min_score']}: {len(pairs)}\n")
        self.stdout.write(f"{'ID_A':>5} {'NAMA_A':<22} {'ID_B':>5} {'NAMA_B':<22} {'SKOR':>7}  KET")
        self.stdout.write("-" * 80)

        rows = []
        for i, j, s in pairs:
            a, b = entries[i], entries[j]
            na, nb = a.nama_merek, b.nama_merek
            ket = "NAMA SAMA" if str(na).strip().lower() == str(nb).strip().lower() else "NAMA BEDA"
            self.stdout.write(f"{a.id:>5} {str(na)[:22]:<22} {b.id:>5} {str(nb)[:22]:<22} {s:>7.4f}  {ket}")
            rows.append([a.id, na, b.id, nb, round(float(s), 4), ket])

        with open(opts["csv"], "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id_a", "nama_a", "id_b", "nama_b", "skor", "keterangan"])
            w.writerows(rows)
        self.stdout.write(f"\nDisimpan ke {opts['csv']}")