# SPEAR: Single-cell Prediction of gene Expression from ATAC-seq Regression

<div align="center">
  <img src="docs/images/spear_logo.png" alt="SPEAR Logo" width="400"/>
</div>

SPEAR (Single-cell Prediction of gene Expression from ATAC-seq Regression) is an end-to-end machine-learning framework for training and evaluating multi-omics integration models. The toolkit focuses on predicting the full RNA gene expression vector for every cell directly from its paired single-cell ATAC accessibility profile, pairing reproducible preprocessing with a modular model zoo, batchable CLI entry points, and plotting notebooks for downstream analysis.

## Key Features

### Project Goals

- Predict the full RNA expression vector for every cell directly from its paired single-cell ATAC accessibility profile.
- Build feature matrices using ±10 kb windows around each gene's transcription start site, binned at 500 bp (40 bins per gene) and recomputed for every run.
- Provide a modular model zoo spanning convolutional, recurrent, transformer, graph-based, gradient boosting (XGBoost, CatBoost), multilayer perceptron, tree ensembles, and linear baselines (Ridge, Elastic Net, Lasso, OLS) so comparative experiments are one CLI flag away.
- Produce test-set diagnostics (scatter plots, per-gene Pearson summaries, epoch histories) while persisting raw predictions for reproducibility and downstream analysis.
- Support reproducible batch execution via Slurm with detailed logging and chunk-aware job arrays.

- **Modal-specific normalization:**
  - ATAC matrices are reindexed to match RNA cell order; by default they are TF–IDF transformed (`tfidf` layer), with alternatives (`counts_per_million`, `log1p_cpm`, or none) available via the `atac_layer` setting.
  - RNA counts are converted to counts-per-million and log-transformed (`log1p_cpm` layer), then scaled with `scanpy.pp.scale` using the same zero-centering guard. A guard prevents double log1p transforms when a normalized layer already exists.
  - Optional `StandardScaler`/`MinMaxScaler` may be applied on the feature and target side based on the run configuration.
- **k-NN smoothing:** Each cell is smoothed by averaging with its k nearest neighbors (default k=19 for `smoothing_k=20`) using PCA-informed nearest-neighbor search to reduce sparsity while maintaining dataset size.
- **Optional pseudobulk aggregation:** PCA-informed, group-aware pooling within each sample when `pseudobulk_group_size > 1`.
- **Group-aware splitting:** 70/15/15 train/val/test splits with `GroupShuffleSplit` keyed by `group_key` (default `sample`; falls back to random when insufficient groups), plus 5-fold cross-validation using `GroupKFold` when possible.
- **Model zoo:** CNN, RNN, LSTM, Transformer, Graph (implicit message passing), PyTorch MLP, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, CatBoost, SVR, Ridge, Elastic Net, Lasso, and OLS. Each model is defined in `spear.models` and accessible through the CLI.
- **Unified diagnostics:** `analysis/spear_results_analysis.ipynb` replaces prior plotting scripts, generating per-gene Pearson summaries, violin plots, top-genes scatter plots, RMSE comparisons, prediction-vs-truth charts, and epoch history curves directly from run outputs.
- **Batch execution:** `jobs/slurm_spear_cellwise_chunked.sbatch` targets CPU nodes, while `jobs/slurm_spear_cellwise_chunked_gpu.sbatch` allocates GPUs for deep models; both expose environment variables for model selection, chunking, hyperparameter overrides, and manifests.

### Data Snapshot

- **ATAC:** 54,301 cells × 192,248 peaks (sparse float32 counts 1–4). Single `mESC` label with sample sizes ranging ~1.8k–10.7k cells.
- **RNA:** 54,301 cells × 32,285 genes (sparse integer counts up to 7,858 UMIs). QC metrics stored in `obs` (total counts, mitochondrial fraction, etc.).
- **Library sizes:** ATAC median 25k peaks/cell (IQR 14k–39k); RNA median 11k UMIs/cell (IQR 7k–17k).
- **Metadata:** Peak identifiers encode genomic coordinates; GTF parsing ensures gene IDs (e.g., `Kmt5b`) resolve consistently across modalities.

### CRISPR KO Replicates

- Do **not** merge CRISPR knockout replicates when preparing AnnData inputs. Each replicate should remain a distinct sample in `obs` (e.g., `obs["replicate"]`) so biological variance is preserved.
- When running pseudobulk or cross-validation, keep replicate identifiers intact; the pipeline assumes replicates were not combined upstream and will treat each as an independent group.

## Repository Layout

```text
analysis/figs/               # Notebook outputs and generated figures
analysis/spear_results_analysis.ipynb
data/                        # Raw AnnData matrices, manifests, references
jobs/slurm_spear_cellwise_chunked.sbatch
jobs/slurm_spear_cellwise_chunked_gpu.sbatch
src/                         # Core Python package (config, data, training, evaluation)
scripts/                     # Minimal helper scripts (data prep only)
todo.md                      # Running task list
```

## Installation

1. Create/activate your environment (Python ≥ 3.10).
2. Install the Python requirements and the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

> Torch and XGBoost wheels can be large on HPC systems—consider using `pip install --no-cache-dir` if disk quotas are tight.

## Running the Pipeline

The CLI exposes all data preparation and model training settings. Basic example (either `spear` or `python -m spear.cli`):

```bash
spear \
  --base-dir "$(pwd)" \
  --models lstm transformer \
  --gene-manifest data/embryonic/manifests/selected_genes_100.csv \
  --epochs 100 \
  --pseudobulk-group-size 20 \
  --device auto
```

Environment / CLI highlights:

- `--gene-manifest` guarantees that every model trains on the same gene subset.
- `--chromosomes genome-wide` (default) processes all annotations; provide a list to restrict loci.
- `--run-name` customises the output directory name.
- `--device` supports `cuda`, `cpu`, or `auto` (prefers CUDA when available; falls back otherwise).
- `--disable-pseudobulk` is a quick toggle to benchmark true single-cell training (equivalent to setting `--pseudobulk-group-size 1`).
- `--atac-layer` lets you swap CPM for alternative ATAC transforms such as `tfidf` or disable normalisation entirely.

For large sweeps, schedule via Slurm:

```bash
MODELS="transformer mlp" \
RUN_NAME=allgenes_meta20_k5 \
GENE_MANIFEST=data/embryonic/manifests/all_genes_annotated.csv \
sbatch --array=1-4 jobs/slurm_spear_cellwise_chunked_gpu.sbatch

# CPU-only baseline (tree ensembles, SVR, etc.)
MODELS="xgboost svr" \
RUN_NAME=allgenes_meta20_k5_cpu \
GENE_MANIFEST=data/embryonic/manifests/all_genes_annotated.csv \
sbatch --array=1-4 jobs/slurm_spear_cellwise_chunked.sbatch
```

The script multiplies models × chunks across the array indices and auto-configures logging paths, environment setup, and manifest wiring.

## Results & Visualization

1. Run the CLI (locally or via Slurm) to generate metrics, predictions, histories, and fitted artifacts.
2. Open `analysis/spear_results_analysis.ipynb` and execute the cells. Adjust `RUN_INCLUDE_GLOBS` at the top of the notebook if you want to focus on specific runs.
3. Generated figures (violin plots, RMSE bars, scatter plots, epoch curves) are written back to `analysis/figs/` and CSV summaries are stored alongside them.

Only per-gene **test-set** Pearson correlations are emphasised in the visualizations. Validation metrics remain available for context (e.g., epoch curves) but are not part of the main comparisons.

## Supported Models

The following identifiers can be supplied to `--models` (and combined arbitrarily):

- `cnn`
- `rnn`
- `lstm`
- `transformer`
- `graph`
- `mlp`
- `random_forest`
- `extra_trees`
- `hist_gradient_boosting`
- `xgboost`
- `catboost`
- `ridge`
- `elastic_net`
- `lasso`
- `ols`
- `svr`

Each mapping to a display label is tracked in `scripts/model_name_lookup.tsv` and consumed by the plotting notebook.
SVR defaults to a linear kernel with configurable hyperparameters via `TrainingConfig.svr_*`.

### Utilities and scripts

- `scripts/preflight_check.py`: quick readiness probe (env, packages, data paths, GTF, SLURM scripts). Run `python scripts/preflight_check.py --help` for options.
- `scripts/preprocess_endothelial.py`: RDS→AnnData conversion and QC for the endothelial dataset (barcode alignment, MT filtering, min genes/cells).

### Optional dependency sets

- Dev tooling: `pip install -r requirements-dev.txt` (ruff, black, pytest).
- Notebook work: `pip install -r requirements-notebook.txt` (JupyterLab, ipykernel).

## Preprocessing Details

1. **AnnData loading:** ATAC and RNA `h5ad` matrices are loaded through `anndata`, reindexed to a common cell ordering, and coerced to dense/sparse formats as required.
2. **Modal layers:**

   - ATAC: Counts-per-million is created by default (`training.atac_layer='counts_per_million'`); alternative transforms (such as TF–IDF) can be requested via configuration.
   - RNA: log1p CPM (`log1p_cpm`) layer computed on demand; if present, double transforms are skipped.

3. **Gene feature extraction:** For each gene, the pipeline sums ATAC counts within ±10 kb windows, binned at 500 bp. Feature matrices are built on demand for each execution.
4. **Expression filtering:** Genes must have at least `min_cells_per_gene` cells above `min_expression` (defaults: 100 cells, 0.0 expression).
5. **k-NN smoothing:** Each cell is smoothed by averaging with its k nearest neighbors (k = `smoothing_k - 1`, default 19) using PCA-informed neighbor search within each split to reduce sparsity while maintaining dataset size.
6. **Pseudobulk (optional):** If `pseudobulk_group_size > 1`, PCA-guided, group-aware pooling within each `group_key` (default `sample`) produces meta-cells of the requested size.
7. **Scaling:** Feature scalers run on the training split; target scaling is skipped automatically when expression values are already log-transformed (set `force_target_scaling=True` to override).
8. **Splitting:** Train/val/test fractions default to 0.70/0.15/0.15 with `GroupShuffleSplit` by `group_key` (fallback to random splits when too few groups).
9. **Cross-validation:** Within the training split, models run 5-fold CV grouped by `group_key` when possible (else shuffled KFold) before fitting on the full training set.

### Future Directions

- **Shared peak matrix for multi-output models:** The current multi-output pipeline builds one wide feature matrix per run by horizontally concatenating each gene’s ±10 kb ATAC bins. At larger gene counts this duplicates peaks, inflates memory, and complicates scaling. A future optimisation is to construct a shared sparse matrix keyed by peak (or genomic bin) once, expose per-gene index views, and update the training/evaluation code to slice into that shared structure. The refactor touches `PeakIndexer`, feature extraction, the dataset dataclasses, and any downstream consumer that assumes a flat concatenation, but it would dramatically shrink memory usage and make scaling to thousands of genes feasible.
- **Cross-dataset benchmarking:** Run the best-performing models on an external endothelial-cell dataset to validate generalisation beyond the current mESC cohort and surface modality-specific quirks early.
- **Feature-pruned retraining:** Retrain the top model using only the top ~1,000 most important features to measure the trade-off between sparsity, compute cost, and predictive power.
- **TSS-window sweep:** Constrain features to lie within X kb of each transcription start site and sweep/optimise X to pinpoint the most informative regulatory window for different model classes.
