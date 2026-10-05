"""Run reproducible experiments, or load a user's BCHW .npy tensor."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import struct
import urllib.request

import numpy as np

from tensor_layout import (bchw_to_matrix, matrix_to_bchw,
                           reconstruction_errors, verify_correspondence)

ROOT = Path(__file__).resolve().parent
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
MNIST_SHA256 = "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"
SEED = 20261005
FIELDS = ["input", "B", "C", "H", "W", "max_abs_error", "mae", "dtype", "seed", "elements", "nonzero_errors", "verified"]


def manual_tensor():
    return np.array([[[[0, 1, 2], [3, 4, 5]],
                      [[6, 7, 8], [9, 10, 11]]]], dtype=np.int32)


def load_mnist(path):
    compressed = path.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != MNIST_SHA256:
        raise ValueError("MNIST checksum mismatch")
    raw = gzip.decompress(compressed)
    magic, count, height, width = struct.unpack(">IIII", raw[:16])
    if (magic, count, height, width) != (2051, 10000, 28, 28):
        raise ValueError("Unexpected MNIST IDX header")
    if len(raw) != 16 + count * height * width:
        raise ValueError("Invalid MNIST payload length")
    tensor = np.empty((2, 1, height, width), dtype=np.uint8)
    for b in range(2):
        for h in range(height):
            for w in range(width):
                tensor[b, 0, h, w] = raw[16 + (b * height + h) * width + w]
    return tensor


def suite(data_dir):
    yield "Manual ordering", manual_tensor(), "n/a"
    yield "MNIST test samples 0-1", load_mnist(data_dir / "t10k-images-idx3-ubyte.gz"), "n/a"
    for C in (3, 8, 16, 32, 64, 128, 256, 500):
        seed = SEED + C
        rng = np.random.Generator(np.random.PCG64(seed))
        shape = (2, C, 32, 32) if C == 3 else (2, C, 13, 17)
        label = "Synthetic RGB" if C == 3 else f"Synthetic fmap C={C}"
        yield label, rng.standard_normal(shape, dtype=np.float32), seed


def run_case(label, tensor, seed, output, index):
    matrix = bchw_to_matrix(tensor)
    B, C, H, W = tensor.shape
    recovered = matrix_to_bchw(matrix, C, H, W)
    verify_correspondence(tensor, matrix, recovered)
    differences, maximum, mae = reconstruction_errors(tensor, recovered)
    nonzero = int(np.count_nonzero(differences))
    if maximum != 0 or mae != 0 or nonzero != 0:
        raise AssertionError(f"Nonzero reconstruction error: {label}")
    np.save(output / "errors" / f"{index:02d}.npy", differences, allow_pickle=False)
    return dict(zip(FIELDS, [label, B, C, H, W, maximum, mae, str(tensor.dtype),
                             seed, tensor.size, nonzero, "PASS"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Load a finite real BCHW tensor from .npy")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--download-mnist", action="store_true", help="Download only when the cached file is absent")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "errors").mkdir(exist_ok=True)
    mnist_file = args.data_dir / "t10k-images-idx3-ubyte.gz"
    if args.download_mnist and not mnist_file.exists():
        args.data_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MNIST_URL, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != MNIST_SHA256:
            raise ValueError("Downloaded MNIST checksum mismatch")
        mnist_file.write_bytes(data)
    cases = [(f"User tensor: {args.input.name}", np.load(args.input, allow_pickle=False), "n/a")] if args.input else suite(args.data_dir)
    rows = [run_case(label, tensor, seed, args.output_dir, i)
            for i, (label, tensor, seed) in enumerate(cases)]
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["Input | B | C | H | W | Max abs error | MAE | Check"]
    for row in rows:
        lines.append(" | ".join(str(row[key]) for key in FIELDS[:7] + ["verified"]))
    lines.append(f"PASS: {len(rows)} experiments; {sum(r['elements'] for r in rows)} elements checked; all errors zero.")
    console = "\n".join(lines) + "\n"
    print(console, end="")
    (args.output_dir / "console_output.txt").write_text(console, encoding="utf-8")
    manifest = {"python": platform.python_version(), "numpy": np.__version__,
                "rng": "PCG64", "base_seed": SEED, "experiments": len(rows),
                "mnist_url": MNIST_URL if not args.input else None,
                "mnist_sha256": MNIST_SHA256 if not args.input else None,
                "results_sha256": hashlib.sha256((args.output_dir / "results.csv").read_bytes()).hexdigest()}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
