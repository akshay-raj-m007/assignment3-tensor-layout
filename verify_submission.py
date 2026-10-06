"""Check that LaTeX, PDF, console, CSV, manifest and saved errors agree."""
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np
import pymupdf

from experiments import MNIST_SHA256, FIELDS
from build_report import tex_escape

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    rows = list(csv.DictReader((ROOT / "results.csv").open(encoding="utf-8", newline="")))
    require(len(rows) == 10, "Expected ten experiment rows")
    require({int(r["C"]) for r in rows} >= {1, 3, 8, 16, 32, 64, 128, 256, 500}, "Missing channel count")
    manifest = json.loads((ROOT / "run_manifest.json").read_text())
    digest = hashlib.sha256((ROOT / "results.csv").read_bytes()).hexdigest()
    require(digest == manifest["results_sha256"], "Manifest/CSV mismatch")
    require(hashlib.sha256((ROOT / "data/t10k-images-idx3-ubyte.gz").read_bytes()).hexdigest() == MNIST_SHA256, "MNIST checksum mismatch")
    console = (ROOT / "console_output.txt").read_text(encoding="utf-8")
    latex = (ROOT / "report.tex").read_text(encoding="utf-8")
    require(digest in latex, "LaTeX CSV checksum mismatch")
    doc = pymupdf.open(ROOT / "report.pdf")
    require(len(doc) == 3, "Report page count")
    result_page = " ".join(unicodedata.normalize("NFKC", doc[1].get_text()).split())
    all_text = " ".join(unicodedata.normalize("NFKC", " ".join(page.get_text() for page in doc)).split())
    require(digest in all_text, "Report CSV checksum mismatch")
    require("[0,1,2,3,4,5,6,7,8,9,10,11]" in all_text.replace(" ", ""), "Missing manual ordering")
    for index, row in enumerate(rows):
        shape = tuple(int(row[k]) for k in ("B", "C", "H", "W"))
        errors = np.load(ROOT / "errors" / f"{index:02d}.npy", allow_pickle=False)
        require(errors.shape == shape, f"Error shape mismatch: {index}")
        require(errors.size == int(row["elements"]), f"Element count mismatch: {index}")
        require(np.isfinite(errors).all(), f"Nonfinite errors: {index}")
        require(float(np.max(np.abs(errors))) == float(row["max_abs_error"]) == 0, "Max error mismatch")
        require(float(np.mean(np.abs(errors))) == float(row["mae"]) == 0, "MAE mismatch")
        require(int(np.count_nonzero(errors)) == int(row["nonzero_errors"]) == 0, "Nonzero error count")
        require(row["verified"] == "PASS", "Experiment not verified")
        expected_console = " | ".join(row[key] for key in FIELDS[:7] + ["verified"])
        require(expected_console in console, f"Console row mismatch: {index}")
        expected_pdf = " ".join(row[key] for key in FIELDS[:7])
        require(expected_pdf in result_page, f"PDF table row mismatch: {index}")
        expected_latex = " & ".join(tex_escape(row[key]) for key in FIELDS[:7]) + r" \\"
        require(expected_latex in latex, f"LaTeX table row mismatch: {index}")
    total = sum(int(row["elements"]) for row in rows)
    require(f"PASS: 10 experiments; {total} elements checked; all errors zero." in console, "Console summary mismatch")
    require(f"{total:,} elements verified" in all_text, "PDF summary mismatch")
    print("PASS: LaTeX, PDF, CSV, console, manifest, MNIST checksum and all 10 difference tensors agree.")


if __name__ == "__main__":
    main()
