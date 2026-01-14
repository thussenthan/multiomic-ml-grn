# SPEAR End-to-End Runbook (Single-cell Prediction of gene Expression from ATAC-seq Regression)

This runbook connects the biological motivation for the mouse embryonic stem cell (mESC) project with the computational workflow implemented in the SPEAR repository. Follow the stages in order when onboarding a new environment, reproducing results, or preparing a public release.

## Orientation

- Scope: infer gene regulatory networks from paired single-cell RNA-seq and ATAC-seq across E7.5 to E8.75 mouse embryogenesis with CRISPR controls.
- Target users: computational biologists operating on the Penn State College of Medicine HPC cluster (Slurm) or a comparable environment.
- Output: model checkpoints, per-gene metrics, aggregate summaries, and narrative analyses ready for publication.

### Biological frame

- mESC differentiation offers a controlled setting to study lineage priming and regulatory switches.
- Multi-omic pairing ensures perturbations in chromatin accessibility can be linked to transcriptional output.
- Replicate structure (two per stage when available) supports variance estimation and biological replicability.

### Computational frame

- Data volume (10x matrices and ATAC fragment files) requires staged downloads and curated AnnData objects.
- Training splits gene targets into HPC-friendly chunks and schedules heterogeneous model families.
- Results consolidation and metadata capture are necessary for reproducibility and comparative modeling.

### Key data products

- Raw GEO downloads and curated AnnData layouts are documented in `docs/mouse_esc_dataset.md`.
- Gene manifests live under `data/embryonic/manifests/` and define target scopes for each run.
- Reference annotations are kept in `data/reference/`.

## Stage 1 - Bootstrap Environment

### Stage 1 - Practical steps

- Clone the repository into `$HOME` or project scratch.
- Install Conda or Mamba if not already available.
- Create the environment: `conda env create -f environment.yml` (or `mamba env create ...`).
- Activate your conda environment and install the project in editable mode (`pip install -e .`) so local modifications are picked up.
- Optional: install dev tools (`pip install -r requirements-dev.txt`) and notebook tooling (`pip install -r requirements-notebook.txt`) if you plan to lint, test, or run notebooks locally.

### Stage 1 - Computer science perspective

- Pinning to the provided environment file locks Python, CUDA, and ML library versions needed for GPU training and consistent serialization.
- Editable installs ensure CLI entry points and module imports resolve to the current working tree, simplifying iterative development.

### Stage 1 - Biology perspective

- Stable environments reduce the risk of numerical drift in downstream metrics, allowing direct biological comparisons with published results.
- Maintaining the same dependency stack as the original analysis preserves behavior of preprocessing routines that enforce biologically motivated QC filters.

## Stage 2 - Acquire and Stage Data

### Stage 2 - Practical steps

- Review `docs/mouse_esc_dataset.md` for provenance and sample inventory.
- Download raw data from GEO accession GSE205117 and place in `data/embryonic/raw/`.
- Preprocess data using scripts in `scripts/` to generate AnnData files in `data/embryonic/processed/`.
- Ensure reference GTF files are present in `data/reference/`.

### Stage 2 - Computer science perspective

- The download script is idempotent: files already present are skipped, preventing accidental re-transfer of large archives.
- Consistent directory layout enables automation in Snakemake-style pipelines and simple globbing in analysis scripts.
- Validating AnnData objects guards against schema mismatches that would break downstream ingestion.

### Stage 2 - Biology perspective

- Confirming each replicate directory contains both RNA and ATAC modalities preserves the paired design necessary for integrative inference.
- Refreshing processed matrices after raw updates avoids mixing batches generated with different cell filters or peak calling parameters.

## Stage 3 - Understand Configuration Surface

### Stage 3 - Practical steps

- Read `docs/pipeline_overview.md` to understand component boundaries (data loaders, feature builders, model zoo, reporting).
- Consult `docs/config_reference.md` for every CLI flag and environment variable supported by `src`.
- Determine the gene manifest(s), chromosome scope, window size, and training overrides for your planned run; record these in `todo.md` or a run sheet.
- For SVR runs, note that `TrainingConfig` exposes `svr_kernel`, `svr_C`, `svr_epsilon`, `svr_max_iter`, and `svr_tol` with defaults documented in `docs/config_reference.md`.

### Stage 3 - Computer science perspective

- Many script parameters have sensible defaults but interact (e.g., `--multi-output` with chunk count); reviewing the reference prevents invalid combinations.
- Knowing the configuration space upfront facilitates reproducibility by enabling exact command reconstruction from logged YAML/JSON artifacts.

### Stage 3 - Biology perspective

- Deciding on gene subsets (pan-cellular vs lineage-specific) and genomic windows frames the biological hypotheses each run can test.
- Clarifying perturbation cohorts (wild-type vs CRISPR) ahead of time ensures downstream comparisons remain interpretable.

## Stage 4 - Internalize Pipeline Architecture

### Conceptual summary

- Feature engineering unites motif scans, accessibility windows, and sequence-derived covariates to represent regulatory potential.
- Training operates per-chunk, where each chunk is a subset of target genes; Slurm arrays map chunk/model pairs to jobs for scalability.
- Model families span deep learning (CNN, RNN, LSTM, Transformer, MLP) and classical approaches (XGBoost, Random Forest, Elastic Net) to capture both nonlinear and interpretable regimes.
- Results artifacts are organized hierarchically by run name, model, and chunk.

### Why it matters

- Understanding the architecture informs how to interpret partial failures (for example, a single chunk timing out only affects the associated gene subset).
- Awareness of model diversity guides ensemble strategies and selection of biology-facing summaries (feature importances vs attention maps).

## Stage 5 - Prime Output and Metadata Directories

### Stage 5 - Practical steps

- Ensure output directories exist (`mkdir -p` as needed).
- Prepare a dedicated log directory if cluster policy requires absolute paths distinct from the repo.
- Confirm that storage quotas can accommodate model checkpoints, particularly for Transformer runs.

### Stage 5 - Computer science perspective

- Pre-creating directories avoids race conditions in batch jobs and keeps Slurm output organized.
- Planning storage usage prevents silent failures caused by quota exhaustion mid-training.

### Stage 5 - Biology perspective

- Organized outputs streamline later aggregation into figures and tables for manuscripts.
- Maintaining log lineage enables provenance tracking when reported biological insights rely on specific training runs.

## Stage 6 - Run Local Smoke Tests

### Stage 6 - Practical steps

- Execute a CPU-only smoke run (via `spear` or module form):  
  `spear --models mlp --gene-manifest data/embryonic/manifests/selected_genes_10.csv --device cpu --k-folds 2 --epochs 2 --run-name dev_smoke_local`
- Confirm outputs are generated successfully.
- Inspect logs for import errors, missing data references, or serialization issues.
- You can also run `python scripts/preflight_check.py` to validate environment, package availability, data paths (AnnData/GTF), and SLURM scripts before queueing jobs.

### Stage 6 - Computer science perspective

- Early detection of dependency or path problems saves GPU queue time and ensures serialization schemas match expectations.

### Stage 6 - Biology perspective

- Even miniature runs validate that cell metadata lines up with gene manifests (e.g., no empty matrices), avoiding biological misinterpretations later.

## Stage 7 - Submit Cluster Jobs

### Stage 7 - Practical steps

- For GPU-capable deep models:  
  `GENE_MANIFEST="$PWD/data/embryonic/manifests/selected_genes_1000.csv" MODELS="cnn rnn lstm transformer mlp" RUN_NAME=spear_gpu_full EXTRA_ARGS="--epochs 200 --batch-size 256 --k-folds 5 --train-fraction 0.7 --val-fraction 0.15 --test-fraction 0.15" sbatch --array=1-60 jobs/slurm_spear_cellwise_chunked_gpu.sbatch`
- For CPU ensembles:  
  `GENE_MANIFEST="$PWD/data/embryonic/manifests/selected_genes_1000.csv" MODELS="xgboost random_forest hist_gradient_boosting elastic_net ridge" RUN_NAME=spear_cpu_full EXTRA_ARGS="--k-folds 5" sbatch --array=1-60 jobs/slurm_spear_cellwise_chunked.sbatch`
- Adjust partitions, memory, and array bounds according to cluster limits and `len(MODELS) * CHUNK_TOTAL`.

### Stage 7 - Computer science perspective

- Slurm arrays map deterministically to `(model, chunk)` pairs; log files encoded with array indices make troubleshooting parallel jobs tractable.
- Explicit environment variables keep submission commands self-documenting and reusable in automation scripts.

### Stage 7 - Biology perspective

- Running diverse model classes provides complementary evidence of regulatory influence (e.g., nonlinear vs linear importance patterns).
- Pairing runs with specific manifests (e.g., endothelial vs global genes) focuses the analysis on biologically coherent questions.

## Stage 8 - Monitor Execution and Validate Intermediate Outputs

### Stage 8 - Practical steps

- Track queues with `sq -u $USER` and tail log files under `output/logs`.
- Spot-check GPU utilization via `srun --pty nvidia-smi` when allowed.
- Confirm each chunk writes `metrics_per_gene.csv`, `summary_metrics.csv`, and `run_configuration.json`.
- Retry failed array indices after addressing the root cause.

### Stage 8 - Computer science perspective

- Monitoring ensures convergence issues, memory exhaustion, or library mismatches are caught promptly.
- Captured configuration snapshots serve as ground truth when regenerating results or sharing with collaborators.

### Stage 8 - Biology perspective

- Mid-run checks verify that metrics fall within expected biological ranges (e.g., correlation coefficients not trivially zero), preventing wasted compute on pathological settings.

## Stage 9 - Aggregate, Interpret, and Visualize

### Stage 9 - Practical steps

- Use scripts in `scripts/` (e.g., `combine_chunk_results.py`) or notebooks under `analysis/` to merge chunk outputs.
- Generate model comparison plots and feature importance summaries; store figures in `analysis/figs`.
- Document findings in Markdown or notebooks, referencing run names and configuration hashes.

### Stage 9 - Computer science perspective

- Consistent aggregation pipelines avoid manual copy/paste errors and support reproducible figure regeneration.
- Persisting intermediate tables enables future statistical testing or meta-analysis without re-running heavy jobs.

### Stage 9 - Biology perspective

- Examine whether top regulatory features align with known developmental regulators, CRISPR perturbation expectations, or spatial gradients.
- Compare accessibility-weighted features against gene expression shifts to propose mechanistic hypotheses.

## Stage 10 - Ensure Reproducibility and Prepare for Release

### Stage 10 - Practical steps

- Update documentation (README, dataset notes, this runbook) with any deviations or newly supported workflows.
- Remove or mask institutional identifiers before sharing publicly.
- Package final results with metadata: run name, date, commit hash, environment spec, and data provenance.
- Optionally, register key outputs in persistent storage (lab archive or institutional repository).

### Stage 10 - Computer science perspective

- Capturing commit hashes and environment files enables deterministic reruns, a prerequisite for confident software releases.
- Reviewing scripts for hard-coded paths or user-specific assumptions prevents portability issues.

### Stage 10 - Biology perspective

- Contextual notes describing biological interpretation, quality thresholds, and open questions transform raw metrics into actionable insights for collaborators.
- Proper provenance documentation supports peer review, future integrative analyses, and compliance with data-sharing policies.

## Additional References

- Dataset specifics and download automation: `docs/mouse_esc_dataset.md`
- Configuration dictionary: `docs/config_reference.md`
- Pipeline component overview: `docs/pipeline_overview.md`
- Job submission templates: `jobs/`
