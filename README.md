# AI Accelerator Design - Assignment 3

Explicit BCHW/NCHW to B x (C*H*W) conversion and reconstruction using Python loops and index arithmetic.

- [Python conversion source](tensor_layout.py)
- [Three-page PDF report](report.pdf)
- [Measured results](results.csv)
- [Setup, reproduction commands, provenance and complete requirement checklist](EVALUATOR.md)

All 10 experiments passed, covering C = 1, 3, 8, 16, 32, 64, 128, 256, 500 plus the manual C=2 example. Every maximum absolute error and MAE is 0.0. The suite verifies 451,492 elements. Sixteen automated tests additionally check ordering, inverse arithmetic and error detection.

The MNIST test archive is included for offline reproduction. RGB-shaped and higher-channel feature maps are honestly labeled synthetic.
