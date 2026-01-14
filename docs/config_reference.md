# SPEAR Configuration Reference

This document outlines every user-facing knob in SPEAR (Single-cell Prediction of gene Expression from ATAC-seq Regression), implemented in the `spear` package, including CLI flags, training defaults, model hyperparameters, and Slurm environment variables.

## Command Line Interface

The CLI entrypoint is `spear` (or `python -m spear.cli`). The table below lists all flags and
their defaults.

| Flag                          | Type      | Default                                                | Description                                                                                              |
| ----------------------------- | --------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `--base-dir`                  | path      | current working directory                              | Project root used to resolve data and output paths.                                                      |
| `--genes`                     | list[str] | `None`                                                 | Explicit list of genes to process. Overrides random sampling.                                            |
| `--gene-manifest`             | path      | `None`                                                 | Path to newline-/CSV-delimited gene manifest.                                                            |
| `--chromosomes`               | list[str] | `None` (auto selects `chr19` in multi-output)          | Filter genes by chromosome. Accepts `genome-wide`/`all`.                                                 |
| `--max-genes`                 | int       | `None`                                                 | Hard cap on genes processed in per-gene mode. Multi-output uses full filtered set unless explicitly set. |
| `--models`                    | list[str] | `['cnn','rnn','lstm','mlp','xgboost','random_forest']` | Replace the default model roster (graph, catboost, etc. can be specified explicitly).                    |
| `--extra-models`              | list[str] | `[]`                                                   | Extend the model roster without replacing defaults.                                                      |
| `--k-folds`                   | int       | `5`                                                    | Number of CV folds.                                                                                      |
| `--train-fraction`            | float     | `0.70`                                                 | Train split proportion (per gene mode).                                                                  |
| `--val-fraction`              | float     | `0.15`                                                 | Validation split proportion.                                                                             |
| `--test-fraction`             | float     | `0.15`                                                 | Test split proportion.                                                                                   |
| `--window-bp`                 | int       | `10000`                                                | ATAC window size around each TSS.                                                                        |
| `--bin-size-bp`               | int       | `500`                                                  | ATAC bin size (peak resolution).                                                                         |
| `--scaler`                    | enum      | `standard`                                             | Feature scaler (`standard`, `minmax`, `none`).                                                           |
| `--target-scaler`             | enum      | `standard`                                             | Target scaler (`standard`, `minmax`, `none`).                                                            |
| `--epochs`                    | int       | `100`                                                  | Training epochs for neural models.                                                                       |
| `--learning-rate`             | float     | `1e-3`                                                 | Optimizer learning rate (torch models).                                                                  |
| `--batch-size`                | int       | `256`                                                  | Mini-batch size (torch models).                                                                          |
| `--pseudobulk-group-size`     | int       | `20`                                                   | Cells per pseudobulk neighborhood.                                                                       |
| `--pseudobulk-pca-components` | int       | `10`                                                   | PCA components for pseudobulk grouping.                                                                  |
| `--disable-pseudobulk`        | flag      | `False`                                                | Shortcut to set `pseudobulk_group_size=1`.                                                               |
| `--device`                    | enum      | `cuda`                                                 | Run device (`cuda`, `cpu`, `auto`).                                                                      |
| `--atac-layer`                | enum      | `counts_per_million`                                   | ATAC normalization (`counts_per_million`, `tfidf`, `none`).                                              |
| `--run-name`                  | str       | Timestamped string                                     | Output run directory name.                                                                               |
| `--chunk-index`               | int       | `0`                                                    | Gene chunk index (zero-based).                                                                           |
| `--chunk-total`               | int       | `1`                                                    | Number of gene chunks.                                                                                   |
| `--config-json`               | path      | `None`                                                 | Load pipeline configuration from JSON file.                                                              |
| `--multi-output`              | flag      | `False`                                                | Enable cell-wise multi-output regression.                                                                |
| `--rf-n-estimators`           | int       | `None` (model-specific default)                        | Override random forest tree count.                                                                       |
| `--rf-max-depth`              | int       | `None`                                                 | Override maximum depth.                                                                                  |
| `--rf-min-samples-leaf`       | int       | `None`                                                 | Override leaf size.                                                                                      |
| `--rf-max-features`           | float/str | `None`                                                 | Override feature fraction.                                                                               |
| `--rf-bootstrap`              | bool      | `None`                                                 | Force bootstrap sampling on/off.                                                                         |

## Training Configuration Defaults

Values below come from `TrainingConfig` and apply unless overridden via CLI or JSON.

| Parameter                   | Default                        | Notes                                                                      |
| --------------------------- | ------------------------------ | -------------------------------------------------------------------------- |
| `window_bp`                 | `10000`                        | +/- bp around TSS.                                                         |
| `bin_size_bp`               | `500`                          | ATAC bin resolution.                                                       |
| `k_folds`                   | `5`                            | Cross-validation folds.                                                    |
| `train_fraction`            | `0.70`                         | Per-gene mode train split.                                                 |
| `val_fraction`              | `0.15`                         | Per-gene mode validation split.                                            |
| `test_fraction`             | `0.15`                         | Per-gene mode test split.                                                  |
| `batch_size`                | `256`                          | Torch mini-batch size.                                                     |
| `epochs`                    | `100`                          | Torch training epochs.                                                     |
| `learning_rate`             | `1e-3`                         | Adam learning rate.                                                        |
| `weight_decay`              | `1e-5`                         | Adam weight decay.                                                         |
| `early_stopping_patience`   | `10`                           | Epoch patience on validation loss.                                         |
| `random_state`              | `42`                           | RNG seed for reproducibility.                                              |
| `device_preference`         | `cuda`                         | Preferred compute device.                                                  |
| `scaler`                    | `standard`                     | Feature scaling (set `none` to disable).                                   |
| `min_cells_per_gene`        | `100`                          | Minimum expressing cells per gene (per-gene mode).                         |
| `min_expression`            | `0.0`                          | Raw expression threshold.                                                  |
| `log1p_transform`           | `False`                        | Additional log1p on targets if raw layer selected.                         |
| `target_scaler`             | `standard`                     | Target scaling (set `none` to disable).                                    |
| `force_target_scaling`      | `False`                        | Apply target scaling even when targets are already log-transformed.        |
| `enable_smoothing`          | `True`                         | Whether to apply k-NN smoothing within each split.                         |
| `smoothing_k`               | `20`                           | Neighborhood size for smoothing (use 1 to disable).                        |
| `smoothing_pca_components`  | `10`                           | PCA components for smoothing neighbor search.                              |
| `pseudobulk_group_size`     | `1`                            | Cells per pseudobulk aggregate (1 disables pooling).                       |
| `pseudobulk_pca_components` | `10`                           | PCA dims for pseudobulk neighborhood search.                               |
| `min_expression_fraction`   | `0.10`                         | Fraction of cells expressing a gene for multi-output sampling.             |
| `rf_n_estimators`           | `None`                         | Falls back to model defaults below.                                        |
| `rf_max_depth`              | `None`                         | Unlimited depth when unset.                                                |
| `rf_min_samples_leaf`       | `None`                         | Model default when unset.                                                  |
| `rf_max_features`           | `None`                         | Model default when unset.                                                  |
| `rf_bootstrap`              | `None`                         | Model default when unset.                                                  |
| `svr_kernel`                | `linear`                       | SVR kernel (`linear`, `rbf`, etc.).                                        |
| `svr_C`                     | `1.0`                          | SVR regularization strength.                                               |
| `svr_epsilon`               | `0.1`                          | Epsilon-insensitive loss parameter.                                        |
| `svr_max_iter`              | `50000`                        | Maximum SVR iterations.                                                    |
| `svr_tol`                   | `1e-4`                         | SVR solver tolerance.                                                      |
| `track_history`             | `True`                         | Record training curves.                                                    |
| `history_metrics`           | `['mse','pearson','spearman']` | Metrics tracked per epoch.                                                 |
| `group_key`                 | `'sample'`                     | obs column used to group splits and CV folds.                              |
| `atac_layer`                | `'counts_per_million'`         | ATAC normalization layer (`counts_per_million`, `tfidf`, etc., or `None`). |
| `rna_expression_layer`      | `'log1p_cpm'`                  | RNA normalization layer.                                                   |

## Paths and Outputs

`PathsConfig.from_base(base_dir)` resolves the following locations relative to the base directory:

| Path Attribute | Default Location                                 |
| -------------- | ------------------------------------------------ |
| `atac_path`    | `data/embryonic/processed/combined_ATAC_qc.h5ad` |
| `rna_path`     | `data/embryonic/processed/combined_RNA_qc.h5ad`  |
| `gtf_path`     | `data/reference/GCF_000001635.27_genomic.gtf`    |
| `output_dir`   | `output/results`                                 |
| `logs_dir`     | `output/logs`                                    |
| `figures_dir`  | `analysis/figs`                                  |

Override paths via CLI flags (`--atac-path`, `--rna-path`, `--gtf-path`) or by supplying a custom JSON config to the pipeline.

## Model Defaults

### Torch Models

| Model         | Architecture Summary                                                                                                                            |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `cnn`         | 1D CNN with 3 conv blocks (32/64/128 channels), adaptive pooling, 512-unit dense head, dropout 0.2.                                             |
| `rnn`         | Conv down-sampling followed by RNN (`hidden_size=96`, `num_layers=1`), dense head with dropout 0.2.                                             |
| `lstm`        | Same conv front-end as RNN, `hidden_size=128` LSTM, dense head with dropout 0.2.                                                                |
| `transformer` | Conv projection to 128 channels, adaptive pooling, transformer encoder (`embed_dim=128`, `num_layers=2`, `num_heads<=8`), dense head with GELU. |
| `graph`       | Implicit 1D graph message-passing network that chunkifies ATAC bins, applies learned edge weights, and aggregates through residual MLP layers.  |
| `mlp`         | Fully connected stack: 256→256→128 with LayerNorm + ReLU + dropout 0.2, output layer sized to target dimension.                                 |

Torch optimizers use `Adam(lr=1e-3, weight_decay=1e-5)` with automatic mixed precision when CUDA is available.

### Scikit-learn / XGBoost Models

| Model                    | Default Hyperparameters                                                                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `svr`                    | `kernel='linear'`, `C=1.0`, `epsilon=0.1`, `max_iter=50_000`, `tol=1e-4`; multi-output uses `MultiOutputRegressor`. Configurable via `TrainingConfig.svr_*`. |
| `xgboost`                | `n_estimators=800`, `max_depth=6`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `tree_method='hist'`.                    |
| `catboost`               | `iterations=1200`, `depth=6`, `learning_rate=0.03`, `loss_function='RMSE'`, auto CPU/GPU detection, verbose disabled.                                        |
| `random_forest`          | `n_estimators=600`, `min_samples_leaf=2`, `max_features=None`, `bootstrap=True`, `oob_score=True`.                                                           |
| `extra_trees`            | `n_estimators=800`, unlimited depth, `min_samples_leaf=1`, `bootstrap=False`.                                                                                |
| `hist_gradient_boosting` | `learning_rate=0.05`, `max_depth=6`, `max_iter=600`, `max_leaf_nodes=64`, `min_samples_leaf=20`, `l2_regularization=1e-3`.                                   |
| `ridge`                  | `alpha=1.0`, with `StandardScaler`.                                                                                                                          |
| `elastic_net`            | Single-target: `alpha=0.05`, `l1_ratio=0.5`; multi-target: `alpha=0.05`, `l1_ratio=0.3`, both with `StandardScaler`.                                         |
| `lasso`                  | Single-target: `alpha=0.05`; multi-target: `alpha=0.05` via `MultiTaskLasso`; both include `StandardScaler`.                                                 |
| `ols`                    | Ordinary least squares (`LinearRegression`) preceded by `StandardScaler`.                                                                                    |

## Slurm Environment Variables

Both job scripts (`jobs/slurm_spear_cellwise_chunked.sbatch` for CPU, `jobs/slurm_spear_cellwise_chunked_gpu.sbatch` for GPU) honor the following variables:

| Variable                    | Default                                                               | Purpose                                                                       |
| --------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `MODELS`                    | GPU: `"cnn rnn lstm transformer mlp"`; CPU: classifier list in script | Space-delimited model roster for the array.                                   |
| `CHUNK_TOTAL`               | `1`                                                                   | Total gene chunks to process. Array size should cover `MODELS * CHUNK_TOTAL`. |
| `GENE_MANIFEST`             | empty                                                                 | Absolute path to manifest passed as `--gene-manifest`.                        |
| `GENOME_SCOPE`              | `genome-wide`                                                         | Value for `--chromosomes` (accepts `all`, chromosome list).                   |
| `RUN_NAME`                  | Autogenerated                                                         | Base run name; script appends chunk/model identifiers.                        |
| `PSEUDOBULK_GROUP_SIZE`     | empty                                                                 | Forwards to `--pseudobulk-group-size`.                                        |
| `PSEUDOBULK_PCA_COMPONENTS` | empty                                                                 | Forwards to `--pseudobulk-pca-components`.                                    |
| `EPOCHS`                    | empty                                                                 | Forwards to `--epochs`.                                                       |
| `BATCH_SIZE`                | empty                                                                 | Forwards to `--batch-size`.                                                   |
| `LEARNING_RATE`             | empty                                                                 | Forwards to `--learning-rate`.                                                |
| `K_FOLDS`                   | empty                                                                 | Forwards to `--k-folds`.                                                      |
| `TRAIN_FRACTION`            | empty                                                                 | Forwards to `--train-fraction`.                                               |
| `VAL_FRACTION`              | empty                                                                 | Forwards to `--val-fraction`.                                                 |
| `TEST_FRACTION`             | empty                                                                 | Forwards to `--test-fraction`.                                                |
| `SCALER`                    | empty                                                                 | Forwards to `--scaler`.                                                       |
| `TARGET_SCALER`             | empty                                                                 | Forwards to `--target-scaler`.                                                |
| `EXTRA_ARGS`                | empty                                                                 | Additional CLI flags, e.g. `"--multi-output"`.                                |
| `MULTI_OUTPUT`              | `1` (GPU) / `1` (CPU script expects)                                  | Enables cell-wise mode when truthy.                                           |
| `BASE_DIR`                  | `$PWD`                                                                | Project root for the run.                                                     |
| `ENV_NAME`                  | `spear_env`                                                           | Conda environment to activate.                                                |
| `GENOME_SCOPE`              | `genome-wide`                                                         | Controls `--chromosomes`.                                                     |

These variables are optional; unset values fall back to script defaults shown above.
