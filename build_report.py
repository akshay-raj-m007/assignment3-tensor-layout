"""Generate the three-page report directly from measured results.csv."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent
NAVY = (0.08, 0.17, 0.28)
TEAL = (0.0, 0.40, 0.44)
GRAY = (0.30, 0.35, 0.40)


def text(page, x, y, value, size=10, font="helv", color=NAVY):
    page.insert_text((x, y), value, fontsize=size, fontname=font, color=color)


def paragraph(page, y, value, height=65, size=10):
    remaining = page.insert_textbox(pymupdf.Rect(44, y, 551, y + height), value,
                                    fontsize=size, fontname="helv", color=NAVY, lineheight=1.45)
    if remaining < 0:
        raise ValueError(f"Report text overflow: {value[:60]}")


def heading(page, y, value):
    text(page, 44, y, value, 14, "hebo", TEAL)


def code(page, y, value, height):
    page.draw_rect(pymupdf.Rect(44, y - 6, 551, y + height), fill=(0.95, 0.97, 0.98), color=None)
    remaining = page.insert_textbox(pymupdf.Rect(55, y, 540, y + height), value,
                                    fontsize=9, fontname="cour", color=NAVY, lineheight=1.25)
    if remaining < 0:
        raise ValueError("Pseudocode overflow")


def new_page(doc, number, title):
    page = doc.new_page(width=595, height=842)
    page.draw_rect(pymupdf.Rect(0, 0, 595, 8), fill=TEAL, color=None)
    text(page, 44, 39, "IIT TIRUPATI  /  AI ACCELERATOR DESIGN", 9, "hebo", GRAY)
    text(page, 44, 72, title, 23, "hebo")
    page.draw_line((44, 792), (551, 792), color=(0.8, 0.85, 0.87))
    text(page, 44, 812, "Assignment 3  |  BCHW tensor layout  |  05 October 2026", 8, color=GRAY)
    text(page, 527, 812, f"{number} / 3", 8, color=GRAY)
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "report.pdf")
    parser.add_argument("--student", default="")
    parser.add_argument("--roll", default="")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.results.open(encoding="utf-8", newline="")))
    if len(rows) != 10 or any(r["verified"] != "PASS" for r in rows):
        raise ValueError("Report requires the complete verified ten-case suite")
    digest = hashlib.sha256(args.results.read_bytes()).hexdigest()
    manifest = json.loads((args.results.parent / "run_manifest.json").read_text())
    if digest != manifest["results_sha256"]:
        raise ValueError("CSV differs from the measured run manifest")
    doc = pymupdf.open()
    doc.set_metadata({"title": "Assignment 3 - BCHW Tensor Layout", "subject": "Explicit indexing and measured reconstruction error",
                      "author": args.student or "", "creator": "build_report.py"})
    page = new_page(doc, 1, "Indexing and reconstruction")
    identity = "  |  ".join(v for v in [args.student, f"Roll no: {args.roll}" if args.roll else ""] if v)
    if identity:
        text(page, 44, 96, identity, 10)
    paragraph(page, 112, "Objective: map I[B,C,H,W] to F[B,C*H*W], then reconstruct I_hat. The batch index is preserved. Within each batch, width varies fastest, followed by height and channel. This gives a linear sequence suitable for tensor buffers and accelerator data transfers.")
    heading(page, 197, "1. Forward mapping")
    code(page, 214, "F = allocate(B, C*H*W, same dtype as I)\nfor b in 0..B-1, c in 0..C-1:\n    for h in 0..H-1, w in 0..W-1:\n        k = (c*H + h)*W + w\n        F[b,k] = I[b,c,h,w]", 88)
    heading(page, 333, "2. Reverse mapping")
    code(page, 350, "I_hat = allocate(B, C, H, W, same dtype as F)\nfor b in 0..B-1, k in 0..C*H*W-1:\n    c = k // (H*W)\n    r = k % (H*W)\n    h = r // W;  w = r % W\n    I_hat[b,c,h,w] = F[b,k]", 103)
    paragraph(page, 470, "Euclidean division makes the mapping bijective: k = c*(H*W) + h*W + w, with 0 <= h < H and 0 <= w < W. Both conversions take O(B*C*H*W) time and allocate one output tensor. The reverse operation requires the original C, H and W as metadata.", 64)
    heading(page, 564, "3. Manual example: B=1, C=2, H=2, W=3")
    code(page, 581, "I[0,0,:,:] = [[0,1,2], [3,4,5]]\nI[0,1,:,:] = [[6,7,8], [9,10,11]]\nF[0,:] = [0,1,2,3,4,5,6,7,8,9,10,11]\n\n(0,0,0,2) -> k=(0*2+0)*3+2=2  -> F[0,2]=2\n(0,0,1,0) -> k=(0*2+1)*3+0=3  -> F[0,3]=3\n(0,1,0,0) -> k=(1*2+0)*3+0=6  -> F[0,6]=6\n(0,1,1,2) -> k=(1*2+1)*3+2=11 -> F[0,11]=11", 133)
    paragraph(page, 734, "Inverse example: k=10 gives c=10//6=1, r=10%6=4, h=4//3=1, w=4%3=1. Therefore F[0,10] returns to I_hat[0,1,1,1]=10.", 43, 9)

    page = new_page(doc, 2, "Experiments and measured errors")
    paragraph(page, 109, "Each run checks every flattened element, reconstructed element and inverse index. The signed difference tensor E = I - I_hat is computed explicitly and saved in errors/00.npy through errors/09.npy in table order. Metrics are calculated over all B*C*H*W elements.", 64)
    code(page, 184, "Emax = max over b,c,h,w of abs(E[b,c,h,w])\nMAE  = sum over b,c,h,w of abs(E[b,c,h,w]) / (B*C*H*W)", 45)
    heading(page, 262, "4. Results (read directly from results.csv)")
    xs = [50, 252, 285, 325, 364, 412, 493]
    page.draw_rect(pymupdf.Rect(44, 281, 551, 307), fill=NAVY, color=None)
    for x, label in zip(xs, ["Input", "B", "C", "H", "W", "Emax", "MAE"]):
        text(page, x, 298, label, 9, "hebo", (1, 1, 1))
    for index, row in enumerate(rows):
        top = 307 + index * 27
        if index % 2 == 0:
            page.draw_rect(pymupdf.Rect(44, top, 551, top + 27), fill=(0.95, 0.97, 0.98), color=None)
        for x, key in zip(xs, ["input", "B", "C", "H", "W", "max_abs_error", "mae"]):
            text(page, x, top + 18, row[key], 9)
    total = sum(int(row["elements"]) for row in rows)
    text(page, 44, 603, f"10 / 10 experiments passed  |  {total:,} elements verified", 13, "hebo", TEAL)
    paragraph(page, 626, "MNIST uses the first two test images, unnormalized uint8 grayscale values. The three-channel input is synthetic RGB-shaped data, not CIFAR. All larger inputs are synthetic float32 feature maps. Synthetic samples use NumPy PCG64, seed = 20261005 + C, and a standard normal distribution.", 74)
    paragraph(page, 720, "All measured Emax and MAE values are 0.0; every saved element-wise difference is zero. No arithmetic is applied to values during either conversion, and dtype is preserved, so exact reconstruction is expected.", 55)

    page = new_page(doc, 3, "Validation and discussion")
    heading(page, 119, "5. Verification")
    paragraph(page, 138, "The 16 automated unit tests cover handwritten expected ordering, B>1, non-square H/W, singleton dimensions, all nine requested channel counts and four numeric dtypes, noncontiguous inputs, index bijection, input preservation, invalid inputs and explicit-loop implementation. Deliberate matrix permutations and reconstruction corruption must be detected.", 83)
    paragraph(page, 235, "A separate nonzero-error test uses unsigned inputs with expected signed differences [-255, 255, 2, -3], Emax=255 and MAE=128.75. This checks the error calculator itself and guards against unsigned subtraction wraparound. Integer values are subtracted before conversion to floating-point metrics.", 69)
    heading(page, 338, "6. Observations and limitations")
    paragraph(page, 357, "Zero error across C=1 through C=500 supports correct channel boundaries and preservation of the batch dimension. The independent ordering check is essential: a mutually consistent pair of wrong permutations could pass reconstruction alone. Quotient/remainder tests establish that every valid linear address has exactly one BCHW coordinate.", 81)
    paragraph(page, 450, "The implementation uses explicit Python loops and is intended to explain address generation, not benchmark accelerator throughput. It supports positive dimensions and finite real integer/floating-point NumPy tensors. Empty axes, NaN, infinity, complex and object values are rejected. Differences and aggregate metrics are stored as float64; exact integer subtraction precedes storage. No optional library layout-conversion reference is used.", 98)
    heading(page, 580, "7. Reproducibility and sources")
    paragraph(page, 599, f"Tested environment: Python {manifest['python']}, NumPy {manifest['numpy']}. Run python -m unittest -v, python experiments.py, python build_report.py, and python verify_submission.py. EVALUATOR.md provides setup, full provenance, an instruction-by-instruction audit and submission details.", 65)
    paragraph(page, 675, "Assignment source: assgn3-tensor-flat.pdf, pages 1-3 (provided by the user). MNIST source: CVDF's public mirror of the MNIST test images. The full archive is included in data/ and verified by SHA-256; no network is needed to repeat the experiments.", 55, 9)
    text(page, 44, 746, "Dataset reference: https://github.com/cvdfoundation/mnist", 9)
    page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(44, 735, 420, 750), "uri": "https://github.com/cvdfoundation/mnist"})
    text(page, 44, 768, f"Results SHA-256: {digest}", 6.8, "cour")
    doc.save(args.output, garbage=4, deflate=True)
    print(f"Created {args.output.name}: {len(doc)} pages; 10 measured rows; CSV SHA-256 {digest}")


if __name__ == "__main__":
    main()
