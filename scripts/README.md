# Utility Scripts

Small helper scripts that sit alongside the core pipeline:

- `select_random_genes.py` – validate expression thresholds and build reusable gene manifests.
- `combine_filtered.py`, `preprocess.py` – data wrangling helpers used prior to training
- `combine_chunk_results.py` – stitch together per-chunk training outputs into unified result folders.

All model diagnostics and plotting have moved to the Jupyter notebook at `analysis/spear_results_analysis.ipynb`, which supersedes the former `plot_*.py` scripts.

Run any remaining script with `python scripts/<name>.py --help` for the available options.
