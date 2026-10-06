# Evaluation and reproduction

This submission implements IIT Tirupati, AI Accelerator Design, Assignment 3, using all three pages of the supplied `assgn3-tensor-flat.pdf` as its specification. The PDF states a deadline of **6 October 2026**. Actual submission to the course portal is the student's responsibility. A name and roll number were not supplied; add them with the report command below if required.

## Files

| File | Purpose |
|---|---|
| `tensor_layout.py` | Explicit forward/reverse algorithms, index helpers, comparison and exhaustive verification |
| `experiments.py` | Reproducible suite, MNIST IDX loader, optional download, user `.npy` input |
| `test_tensor_layout.py` | 16 automated tests, including an AST call audit |
| `results.csv` | Actual measured results and run dimensions |
| `errors/00.npy` through `errors/09.npy` | Full signed element-wise difference tensors, in CSV row order |
| `console_output.txt`, `run_manifest.json` | Captured experiment output, versions, seeds and CSV checksum |
| `report.tex`, `report.pdf`, `build_report.py` | Editable standalone LaTeX, corresponding three-page PDF, and CSV-refresh/compiler driver |
| `latex_build_output.txt` | Actual output of the LaTeX compiler |
| `verify_submission.py` | Cross-check LaTeX/PDF/CSV/console/manifest/difference tensors/data checksum |
| `data/t10k-images-idx3-ubyte.gz` | Original MNIST test image archive, unchanged |
| `requirements.txt` | Pinned Python dependencies: NumPy and PyMuPDF (PDF inspection/verification only) |
| `test_output.txt`, `verification_output.txt` | Output from final validation commands |

## Exact setup and run commands

Use Python 3.12 or newer; this submission was executed with Python 3.14.2. Open a terminal **in this repository**. The bundled dataset makes experiment execution offline; dependency installation may require internet access. No GPU, PyTorch or notebook is needed. Rebuilding the PDF requires a LaTeX compiler. This report was compiled from `report.tex` with **Tectonic 0.17.0**; PyMuPDF only reads/verifies the resulting PDF.

On Windows, install the portable compiler locally (no system installation or PATH modification needed):

```powershell
New-Item -ItemType Directory -Force .tools
Invoke-WebRequest "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip" -OutFile .tools/tectonic.zip
Expand-Archive .tools/tectonic.zip -DestinationPath .tools -Force
```

The downloaded archive SHA-256 is `f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f`. The compiler executable/cache is not committed. The first compilation downloads standard LaTeX packages and fonts. Alternatively, install TeX Live/MiKTeX and put `pdflatex` or `xelatex` on PATH. On Linux/macOS, follow the [official Tectonic installation instructions](https://tectonic-typesetting.github.io/book/latest/installation/) or use an existing TeX distribution. The build script finds these compilers automatically; `--compiler path/to/compiler` overrides the choice.

On the tested Windows host, Tectonic printed a nonfatal Fontconfig configuration message but completed successfully using its TeX fonts. All 17 fonts in the PDF are embedded; the final LaTeX build has no layout-overflow warnings. All three pages were visually inspected.

Windows PowerShell (activation is unnecessary):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe experiments.py
.\.venv\Scripts\python.exe build_report.py
.\.venv\Scripts\python.exe verify_submission.py
```

Linux/macOS:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest -v
.venv/bin/python experiments.py
.venv/bin/python build_report.py
.venv/bin/python verify_submission.py
```

Expected unit-test output: `Ran 16 tests` followed by `OK`. Experiments print the ten rows in `results.csv`, ending with:

```text
PASS: 10 experiments; 451492 elements checked; all errors zero.
```

Expected artifact-check output:

```text
PASS: LaTeX, PDF, CSV, console, manifest, MNIST checksum and all 10 difference tensors agree.
```

For an independent repeat without overwriting submitted measurements:

```powershell
.\.venv\Scripts\python.exe experiments.py --output-dir reproduced
.\.venv\Scripts\python.exe -c "from pathlib import Path; assert Path('results.csv').read_bytes() == Path('reproduced/results.csv').read_bytes(); print('PASS: repeated CSV is byte-identical')"
```

To add your identity to the report (replace the example text):

```powershell
.\.venv\Scripts\python.exe build_report.py --student "Your Name" --roll "Your Roll Number"
.\.venv\Scripts\python.exe verify_submission.py
```

You may also edit `\StudentName` and `\RollNumber` directly near the top of `report.tex`. The script preserves existing identity values unless those flags are supplied. It preserves prose edits and only replaces the two clearly marked generated blocks containing results and metadata.

To edit/compile the LaTeX directly, upload **only `report.tex`** to Overleaf and select pdfLaTeX, or run one of these local compiler options:

```powershell
.\.tools\tectonic.exe --keep-logs --untrusted report.tex
# Alternatively, with a TeX distribution on PATH:
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape report.tex
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape report.tex
```

The LaTeX file contains the full report, equations, pseudocode and measured table; it has no external image, bibliography or included-file dependencies. To update its measured values without a local compiler, run `python build_report.py --tex-only` before uploading to Overleaf. That option updates the source only; compile afterward to update the corresponding PDF. Compilation logs are kept in `latex_build_output.txt` when using the Python build driver. Auxiliary compiler files are ignored by Git.

To load an existing finite real BCHW/NCHW NumPy tensor:

```powershell
.\.venv\Scripts\python.exe experiments.py --input path/to/tensor.npy --output-dir custom_run
```

This writes separate CSV, console, manifest and element-wise error files. The report generator intentionally requires the full ten-case suite. To re-download the bundled dataset only if it is absent, use `experiments.py --download-mnist`. Download/checksum failures are errors: the program never silently relabels synthetic data as MNIST.

## Algorithms and manual ordering

`bchw_to_matrix` allocates `(B,C*H*W)` and uses four nested loops with `k=(c*H+h)*W+w`. `matrix_to_bchw` allocates `(B,C,H,W)` and loops over `(b,k)`, computing `c=k//(H*W)`, `r=k%(H*W)`, `h=r//W`, `w=r%W`. Batch indices stay unchanged. Both copy scalar values without arithmetic or dtype conversion. Each costs O(BCHW) time and O(BCHW) output storage. Original C/H/W must be retained for reconstruction.

For `(B,C,H,W)=(1,2,2,3)`, channel 0 is `[[0,1,2],[3,4,5]]` and channel 1 is `[[6,7,8],[9,10,11]]`. The complete output is `[[0,1,2,3,4,5,6,7,8,9,10,11]]`.

| Source coordinate | Calculation | Destination |
|---|---|---|
| (0,0,0,2) | (0*2+0)*3+2 = 2 | (0,2) |
| (0,0,1,0) | (0*2+1)*3+0 = 3 | (0,3) |
| (0,1,0,0) | (1*2+0)*3+0 = 6 | (0,6) |
| (0,1,1,2) | (1*2+1)*3+2 = 11 | (0,11) |

For inverse offset 10: `c=10//6=1`, `r=10%6=4`, `h=4//3=1`, `w=4%3=1`. Thus `(b,k)=(0,10)` maps to `(0,1,1,1)`.

## Dataset provenance and reproducibility

- **MNIST:** first two images (zero-based indices 0 and 1) of the original 10,000-image test split. Loaded from the IDX bytes into `(2,1,28,28)` using loops. Values remain unnormalized uint8, range 0-255. Labels are not needed and were not downloaded.
- Mirror owner and provenance: [CVDF MNIST README](https://github.com/cvdfoundation/mnist). Archive URL: [test image archive](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz). Downloaded for this assignment on 5 October 2026. The archive is 1,648,877 bytes. SHA-256: `8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6`. Both header and payload length are validated.
- **Synthetic RGB:** `(2,3,32,32)` float32 standard-normal values. This is **not CIFAR** and is not sampled from an image dataset.
- **Synthetic feature maps:** `(2,C,13,17)` for C = 8, 16, 32, 64, 128, 256, 500; float32 standard-normal values. These simulate multi-channel DNN feature-map shapes; they are not outputs of a trained DNN.
- **Manual input:** int32 constants written explicitly in `manual_tensor()`.
- Each synthetic case starts a fresh NumPy `Generator(PCG64(20261005+C))`. Seeds and dtypes are recorded per CSV row. The pinned NumPy version ensures reproducible generated inputs. `run_manifest.json` records actual environment versions and the CSV SHA-256.

## Actual results

| Input | B | C | H | W | Maximum absolute error | MAE |
|---|---:|---:|---:|---:|---:|---:|
| Manual ordering | 1 | 2 | 2 | 3 | 0.0 | 0.0 |
| MNIST test samples 0-1 | 2 | 1 | 28 | 28 | 0.0 | 0.0 |
| Synthetic RGB | 2 | 3 | 32 | 32 | 0.0 | 0.0 |
| Synthetic fmap C=8 | 2 | 8 | 13 | 17 | 0.0 | 0.0 |
| Synthetic fmap C=16 | 2 | 16 | 13 | 17 | 0.0 | 0.0 |
| Synthetic fmap C=32 | 2 | 32 | 13 | 17 | 0.0 | 0.0 |
| Synthetic fmap C=64 | 2 | 64 | 13 | 17 | 0.0 | 0.0 |
| Synthetic fmap C=128 | 2 | 128 | 13 | 17 | 0.0 | 0.0 |
| Synthetic fmap C=256 | 2 | 256 | 13 | 17 | 0.0 | 0.0 |
| Synthetic fmap C=500 | 2 | 500 | 13 | 17 | 0.0 | 0.0 |

All 451,492 differences are zero. The largest case contains 221,000 elements. A nonzero experiment error raises an exception; it cannot be reported as PASS. `verify_correspondence` compares every matrix element against an independent sequential column traversal, every reconstructed element, and both index formulas. Thus a wrong permutation cannot hide behind a successful round trip.

## Requirement-by-requirement audit

All instructional statements on the three PDF pages are accounted for below. Page 1's start/deadline/roll header is administrative; the deadline and identity limitation are recorded above.

| PDF instruction | Evidence |
|---|---|
| p1: understand image/feature-map linear representation and accelerator motivation | Report p1 objective, p3 discussion |
| p1: BCHW/NCHW input and B x (CHW) intermediate; reconstruct BCHW | `tensor_layout.py`: `bchw_to_matrix`, `matrix_to_bchw`; shape/dtype tests |
| p1: follow the lecture mapping and strengthen indexing | Width-fastest mapping specified by user's formula; report p1 and index-bijection test. Lecture notes were not supplied; no unseen lecture content is claimed |
| p1 task 1: load BCHW input | `experiments.py`: `load_mnist`, `--input`; actual MNIST run |
| p1 tasks 2-3: implement both transformations | Explicit scalar assignments and index formulas in the two conversion functions; report pseudocode |
| p1 task 4: compare reconstructed and original tensors | `verify_correspondence`, `reconstruction_errors`; CSV and saved differences |
| p1 task 5: repeat with different channels | All nine suggested C values plus C=2; results table |
| p1-p2: both algorithms from scratch; loops/index arithmetic | Four forward loops and two reverse loops; `test_algorithm_call_allowlist_and_explicit_loops` |
| p1: permitted library use for data loading, generation, storage, individual access | NumPy allocations/RNG/scalar access and standard-library IDX parser; code inspection |
| p1-p2: no reshape/view/flatten/ravel or equivalent conversions in either algorithm | AST test restricts conversion calls to validation, `np.empty`, `range`, `ValueError`; full-module prohibited-call scan; no shape attribute reassignment or indirect conversion calls |
| p2: any library reference must be optional and separate | No library shape-conversion reference is used anywhere in the submitted implementation |
| p2: small manual example and several mappings | Report p1 and manual table above; handwritten expected-output tests |
| p2: Python source in GitHub and PDF GitHub link | Repository submission links at end of this document |
| p2: short PDF with both pseudocodes, manual example, dimensions, errors, discussion | `report.pdf`: p1 algorithms/manual example; p2 dimensions/errors; p3 discussion |
| User follow-up: editable LaTeX and its corresponding PDF | Standalone `report.tex`; `build_report.py` invokes a real LaTeX compiler; `latex_build_output.txt`; LaTeX/PDF table consistency checks |
| p2: realistic image data and wider-channel synthetic feature maps | Real MNIST archive and correctly labeled synthetic cases; provenance above |
| p2: enough small and large channel sizes | C=1,3,8,16,32,64,128,256,500; report p2 |
| p2: specify every B/C/H/W and use practical sizes | Every CSV row and report p2; maximum 221,000 elements per experiment |
| p2: E = I - I_hat element-wise | `reconstruction_errors`; all ten `errors/*.npy`; signed/unsigned error test |
| p2: maximum absolute error and MAE over BCHW | Explicit scalar accumulation in `reconstruction_errors`; known-nonzero metric test |
| p2: expected zero error if copying unchanged | Every measured row is zero; nonzero results would abort the suite |
| p2-p3: table with Dataset/Input, B, C, H, W, Emax and MAE | `results.csv`, report p2, table above; `verify_submission.py` cross-check |
| p2: submit before the deadline | Files prepared for submission; course-portal submission remains the student's action |

Additional user requirements: the handwritten B>1/non-square test, exhaustive inverse arithmetic tests, exact per-index checks, deliberate corruption checks, actual execution logs, pinned setup, and artifact consistency checker provide the requested verification beyond the PDF minimum.

## Limitations

The algorithms accept positive dimensions and finite real numeric NumPy arrays, including noncontiguous inputs. Invalid ranks, zero/negative dimensions, noninteger indices, inconsistent matrix width, NaN/infinity, complex and object data are rejected. Tensor dtype and values are preserved. Signed differences and metrics use float64 storage; arbitrary-precision error reporting for extreme values is outside this experiment's scope. Python loops favor clarity over speed. No accelerator timing or performance claim is made. GitHub publication does not itself submit anything to the course portal.

## Submission links

- [Complete public repository](https://github.com/akshay-raj-m007/assignment3-tensor-layout)
- [Python conversion source](https://github.com/akshay-raj-m007/assignment3-tensor-layout/blob/main/tensor_layout.py)
- [Runnable experiment program](https://github.com/akshay-raj-m007/assignment3-tensor-layout/blob/main/experiments.py)
- [Editable LaTeX report](https://github.com/akshay-raj-m007/assignment3-tensor-layout/blob/main/report.tex)
- [PDF report](https://github.com/akshay-raj-m007/assignment3-tensor-layout/blob/main/report.pdf)

Submit the source/repository and PDF links to the course portal. The repository includes all required local files and the dataset archive; no additional upload is needed to reproduce the suite.
