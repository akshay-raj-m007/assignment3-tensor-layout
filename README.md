# AI Accelerator Design - Assignment 3

Explicit BCHW/NCHW to B x (C*H*W) conversion and reconstruction using Python loops and index arithmetic.

- [Python conversion source](tensor_layout.py)
- [Editable LaTeX report](report.tex)
- [Corresponding three-page PDF](report.pdf)
- [Measured results](results.csv)
- [Setup, reproduction commands, provenance and complete requirement checklist](EVALUATOR.md)

All 10 experiments passed, covering C = 1, 3, 8, 16, 32, 64, 128, 256, 500 plus the manual C=2 example. Every maximum absolute error and MAE is 0.0. The suite verifies 451,492 elements. Sixteen automated tests additionally check ordering, inverse arithmetic and error detection.

The MNIST test archive is included for offline reproduction. RGB-shaped and higher-channel feature maps are honestly labeled synthetic.

Upload `report.tex` alone to Overleaf, or compile it with `tectonic report.tex` / `pdflatex report.tex`. The source is self-contained. `python build_report.py` refreshes its measured results from the CSV and compiles the PDF; it requires a LaTeX compiler. See `EVALUATOR.md` for installation and editing instructions.
