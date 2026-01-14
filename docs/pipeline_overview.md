# SPEAR Overview (Single-cell Prediction of gene Expression from ATAC-seq Regression)

This document summarizes the current end-to-end workflow for SPEAR (Single-cell Prediction of gene Expression from ATAC-seq Regression)—so new contributors can onboard quickly.

## 1. Data Sources and Configuration

- **AnnData inputs:** `combined_ATAC_qc.h5ad` and `combined_RNA_qc.h5ad` (stored in `data/embryonic/processed/`) are loaded through `PathsConfig.from_base(...)`. Cell barcodes are aligned so paired ATAC/RNA observations remain synchronized. Upstream preprocessing (e.g., removing CRISPR KO cells) should be completed before running the pipeline; no in-pipeline filtering is applied.
- **Reference annotations:** The GTF specified in `config.PathsConfig` (defaults to `data/reference/GCF_000001635.27_genomic.gtf`) provides TSS coordinates used in feature extraction.
- **Training configuration:** `TrainingConfig` centralizes hyperparameters such as window size (±10 kb), bin size (500 bp), k-NN smoothing settings (k=20, 10 PCA components), grouped splitting via `group_key` (default: `sample`), split ratios (70/15/15), and history tracking. Each CLI run receives a `run_name` so all outputs land in a dedicated directory.
- **Chromosome filters:** Genome-wide is the default; pass `--chromosomes` (or set `config.chromosomes`) to restrict to specific contigs.
- **Gene manifests:** Runs optionally accept a newline-delimited manifest (`--gene-manifest`). When present, every model consumes the same ordered gene list; otherwise genes can be filtered by chromosome or count thresholds at runtime.

## 2. Gene Selection Logic

1. Parse the GTF to obtain gene IDs/names and TSS information (respecting chromosome filters or explicit gene lists).
2. Validate that requested genes exist in the RNA AnnData matrix and meet expression criteria (`min_cells_per_gene`, `min_expression`, `min_expression_fraction`).
3. Persist selected genes and their expression fractions alongside each run.

## 3. Feature Construction & Normalization

- **ATAC features:** For every gene we aggregate peak counts within ±10 kb windows around the TSS, binned into 500 bp segments (40 bins by default). Feature matrices are recomputed for each run.
- **Modal normalization:**
  - ATAC counts are converted to counts-per-million by default (`training.atac_layer='counts_per_million'`); other transforms such as TF–IDF can be requested via configuration.
  - RNA counts are converted to log1p CPM (`rna_expression_layer="log1p_cpm"`) unless a precomputed layer already exists and are then standardized with `scanpy.pp.scale` using the same sparse-aware settings.
- **Optional scalers:** Standard or MinMax scaling can be applied to features and/or targets depending on `TrainingConfig`.

## 4. k-NN Smoothing

To reduce sparsity while maintaining dataset size, each cell is smoothed by averaging with its k nearest neighbors:

1. Compute a PCA embedding (default 10 components) within each split.
2. For every cell, find its k nearest neighbors (k = `smoothing_k - 1`, default 19) using `NearestNeighbors` on the PCA embedding.
3. Average the cell's features/targets with its neighbors' data to create a smoothed version. The same number of cells is returned (no reduction in dataset size).

## 5. Model Suite

`spear.models.build_model` exposes a broad model zoo so experiments can toggle architectures via CLI flags:

- Neural models: `cnn`, `rnn`, `lstm`, `transformer`, `graph`, `mlp` (PyTorch).
- Tree ensembles & boosting: `random_forest`, `extra_trees`, `hist_gradient_boosting`, `xgboost`, `catboost`.
- Linear baselines: `ridge`, `elastic_net`, `lasso`, `ols`.
- Kernel methods: `svr` (defaults to linear kernel; tune via `TrainingConfig.svr_*`).

Multi-output training (predicting all genes simultaneously) is the default; use `--per-gene` if you want to iterate one gene at a time.

## 6. Training, Evaluation & Metrics

- **Splits:** 70% train, 15% validation, 15% test. Grouped splits are used when `group_key` is set (default: `sample`); otherwise the pipeline falls back to random splits.
- **Cross-validation:** 5-fold CV precedes final model fitting (grouped when enough unique groups exist, otherwise shuffled KFold).
- **Metrics:** MSE, RMSE, MAE, R², Spearman, and Pearson are recorded per split; per-fold CV metrics are also stored.
- **History tracking:** Torch models optionally log per-epoch loss and correlation metrics for train/validation splits.

## 7. Output Layout

Results are organized by run name and model, with metrics, predictions, and diagnostic figures stored in a consistent hierarchy. Logs for each CLI run are written alongside the outputs.

## 8. Job Submission Workflow

- Use `scripts/select_random_genes.py` to produce reproducible manifests when sampling subsets.
- Submit batch jobs with the generalized script:

  ```bash
  MODELS="cnn lstm transformer mlp" \
  RUN_NAME=allgenes_meta20_k5 \
  GENE_MANIFEST=data/embryonic/manifests/all_genes_annotated.csv \
  sbatch --array=1-4 jobs/slurm_spear_cellwise_chunked_gpu.sbatch

  MODELS="xgboost svr random_forest extra_trees hist_gradient_boosting ridge elastic_net lasso ols" \
  RUN_NAME=allgenes_meta20_k5_cpu \
  GENE_MANIFEST=data/embryonic/manifests/all_genes_annotated.csv \
  sbatch --array=1-9 jobs/slurm_spear_cellwise_chunked.sbatch
  ```

- The script maps each array index to a model × chunk combination, activates the configured Conda environment, and forwards all relevant CLI arguments (device, pseudobulk settings, CV folds, etc.).

## 9. Post-run Analysis

1. Verify array tasks completed successfully and check aggregate metrics.
2. Open `analysis/spear_results_analysis.ipynb`, adjust `RUN_INCLUDE_GLOBS` if necessary, and execute the notebook to regenerate per-gene test Pearson summaries, violin plots, RMSE comparisons, prediction-vs-truth scatters, and epoch curves.
3. Commit regenerated figures/CSVs in `analysis/figs/` so published artifacts match the latest experiments.

## 10. Feature Importance Artifacts

- Torch-based multi-output runs (e.g., MLP/transformer) now emit: `feature_importances_mean.csv` (mean/std/median per feature plus TSS-relative metadata), `feature_importances_raw.npz` (compressed stack for reproducibility), `feature_importance_per_gene_summary.csv` (per-gene statistics and correlations vs. TSS distance), `feature_importance_mean.png`, and `feature_importance_vs_tss_distance.png` inside `models/<model>/`.
- These outputs enable downstream correlation studies without re-running attribution; every feature row includes `gene_name`, `relative_start_bp`, and `distance_to_tss_kb` so plots can be regenerated directly from CSV.
- Use `scripts/plot_feature_importance_vs_tss.py <run_dir> --model mlp --output analysis/figs/fi_mlp.png` to assemble a publication-ready panel (top features, importance vs. TSS scatter, per-gene correlation boxplot). The script gracefully skips panels if metadata is missing.

## 11. Utilities and environment notes

- `scripts/preflight_check.py` checks environment, required packages, data paths (AnnData/GTF), and SLURM scripts before launching jobs.
- `scripts/preprocess_endothelial.py` converts endothelial RDS inputs to AnnData with QC matching the embryonic workflow.
- Optional dependency bundles:
  - Dev: `requirements-dev.txt` (ruff, black, pytest)
  - Notebook: `requirements-notebook.txt` (JupyterLab, ipykernel)

SVR hyperparameters (`svr_kernel`, `svr_C`, `svr_epsilon`, `svr_max_iter`, `svr_tol`) are part of `TrainingConfig` with defaults shown in config_reference.

With these steps the repository is ready for publication: preprocessing is documented, all models are accessible via CLI and Slurm, and downstream analysis is consolidated in a single notebook.
