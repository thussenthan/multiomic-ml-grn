
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import ModelConfig, PipelineConfig, PathsConfig, TrainingConfig
from .evaluation import run_pipeline
from .logging_utils import configure_logging, get_logger


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPEAR: Single-cell Prediction of gene Expression from ATAC-seq Regression")
    parser.add_argument(
        "--base-dir",
        default=str(Path.cwd()),
        help="Project root directory (defaults to the current working directory)",
    )
    parser.add_argument("--atac-path", help="Override ATAC AnnData path (h5ad)")
    parser.add_argument("--rna-path", help="Override RNA AnnData path (h5ad)")
    parser.add_argument("--gtf-path", help="Override GTF annotation path")
    parser.add_argument("--genes", nargs="*", help="Specific gene names to model")
    parser.add_argument("--gene-manifest", help="Path to newline-delimited list of gene names to model")
    parser.add_argument("--chromosomes", nargs="*", help="Limit processing to genes on specific chromosomes")
    parser.add_argument("--max-genes", type=int, help="Maximum number of genes to process")
    parser.add_argument("--models", nargs="*", help="Models to evaluate (override defaults)")
    parser.add_argument("--extra-models", nargs="*", help="Additional models to include")
    parser.add_argument("--k-folds", type=int, help="Number of folds for cross-validation")
    parser.add_argument("--train-fraction", type=float, help="Training fraction (default 0.7)")
    parser.add_argument("--val-fraction", type=float, help="Validation fraction (default 0.15)")
    parser.add_argument("--test-fraction", type=float, help="Test fraction (default 0.15)")
    parser.add_argument(
        "--group-key",
        help="AnnData obs column to use for grouped splits (set to 'none' to disable grouped splitting)",
    )
    parser.add_argument("--window-bp", type=int, help="Window around TSS in base pairs (default 10,000)")
    parser.add_argument("--bin-size-bp", type=int, help="Bin size in base pairs (default 500)")
    parser.add_argument("--scaler", choices=["standard", "minmax", "none"], help="Feature scaler")
    parser.add_argument("--target-scaler", choices=["standard", "minmax", "none"], help="Target scaler")
    parser.add_argument(
        "--force-target-scaling",
        action="store_true",
        help="Apply target scaler even when expression values are already log transformed",
    )
    parser.add_argument("--epochs", type=int, help="Training epochs for neural models")
    parser.add_argument("--learning-rate", type=float, help="Learning rate for neural models")
    parser.add_argument("--batch-size", type=int, help="Batch size for neural models")
    parser.add_argument(
        "--pseudobulk-group-size",
        type=int,
        help="Cells per pseudobulk group (pools/averages cells, reducing dataset size); use 1 to disable pooling",
    )
    parser.add_argument(
        "--pseudobulk-pca-components",
        type=int,
        help="Number of PCA components to build pseudobulk neighborhoods",
    )
    parser.add_argument(
        "--disable-pseudobulk",
        action="store_true",
        help="Disable pseudobulk pooling (equivalent to --pseudobulk-group-size 1)",
    )
    parser.add_argument(
        "--smoothing-k",
        type=int,
        help="Neighborhood size for k-NN smoothing (>=1). Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--smoothing-pca-components",
        type=int,
        help="PCA components for k-NN smoothing neighborhoods",
    )
    parser.add_argument(
        "--disable-smoothing",
        action="store_true",
        help="Disable k-NN smoothing of cells",
    )
    parser.add_argument(
        "--resource-sample-seconds",
        type=float,
        help="Interval (in seconds) between resource usage samples (default 60)",
    )
    parser.add_argument(
        "--disable-feature-importance",
        action="store_true",
        help="Disable feature importance even if enabled by default in config",
    )
    parser.add_argument(
        "--feature-importance-samples",
        type=int,
        help="Max samples for feature importance; omit for ALL samples (default: all)",
    )
    parser.add_argument(
        "--feature-importance-batch-size",
        type=int,
        help="Batch size for feature-importance gradient accumulation (default 128)",
    )
    parser.add_argument(
        "--atac-layer",
        choices=["counts_per_million", "tfidf", "log1p_cpm", "none"],
        help="ATAC normalization layer (default tfidf)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Preferred compute device (cuda, cpu, or auto to choose CUDA when available)",
    )
    parser.add_argument("--run-name", help="Optional run name override")
    parser.add_argument("--chunk-index", type=int, default=0, help="Zero-based index of the gene chunk to process")
    parser.add_argument("--chunk-total", type=int, default=1, help="Total number of gene chunks across all jobs")
    parser.add_argument("--config-json", help="Path to configuration JSON file to load")
    parser.add_argument(
        "--per-gene",
        action="store_true",
        help="Run per-gene training (one model per gene) instead of the default cell-wise multi-output mode",
    )
    parser.add_argument(
        "--multi-output",
        action="store_true",
        help="Explicitly enable cell-wise multi-output mode (default unless --per-gene is set)",
    )
    parser.add_argument("--rf-n-estimators", type=int, help="Number of trees for random forest models")
    parser.add_argument("--rf-max-depth", type=int, help="Maximum depth for random forest models")
    parser.add_argument("--rf-min-samples-leaf", type=int, help="Minimum samples per leaf for random forest models")
    parser.add_argument(
        "--rf-max-features",
        help="Maximum features for random forest models (float fraction or keywords like sqrt)",
    )
    parser.add_argument(
        "--rf-bootstrap",
        choices=["true", "false"],
        help="Enable or disable bootstrap sampling for random forest models",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.chunk_total < 1:
        parser.error("--chunk-total must be >= 1")
    if args.chunk_index < 0 or args.chunk_index >= args.chunk_total:
        parser.error("--chunk-index must satisfy 0 <= chunk_index < chunk_total")

    if args.config_json:
        config_path = Path(args.config_json).expanduser().resolve()
        payload = json.loads(config_path.read_text())
        config = _config_from_json(payload)
    else:
        paths = PathsConfig.from_base(args.base_dir)

        def _override_path(current: Path, override: Optional[str], label: str) -> Path:
            if not override:
                return current
            candidate = Path(override).expanduser().resolve()
            if not candidate.exists():
                parser.error(f"{label} not found at {candidate}")
            return candidate

        paths.atac_path = _override_path(paths.atac_path, args.atac_path, "ATAC path")
        paths.rna_path = _override_path(paths.rna_path, args.rna_path, "RNA path")
        paths.gtf_path = _override_path(paths.gtf_path, args.gtf_path, "GTF path")

        training = TrainingConfig()
        if args.k_folds:
            training.k_folds = args.k_folds
        if args.train_fraction:
            training.train_fraction = args.train_fraction
        if args.val_fraction:
            training.val_fraction = args.val_fraction
        if args.test_fraction:
            training.test_fraction = args.test_fraction
        if args.window_bp:
            training.window_bp = args.window_bp
        if args.bin_size_bp:
            training.bin_size_bp = args.bin_size_bp
        if args.scaler:
            training.scaler = None if args.scaler == "none" else args.scaler
        if args.target_scaler:
            training.target_scaler = None if args.target_scaler == "none" else args.target_scaler
        if args.force_target_scaling:
            training.force_target_scaling = True
        if args.group_key is not None:
            training.group_key = None if args.group_key.lower() == "none" else args.group_key
        if args.epochs:
            training.epochs = args.epochs
        if args.learning_rate:
            training.learning_rate = args.learning_rate
        if args.batch_size:
            training.batch_size = args.batch_size
        if args.smoothing_k is not None:
            training.smoothing_k = args.smoothing_k
        if args.smoothing_pca_components is not None:
            training.smoothing_pca_components = args.smoothing_pca_components
        if args.disable_smoothing:
            training.enable_smoothing = False
        if args.pseudobulk_group_size is not None:
            training.pseudobulk_group_size = args.pseudobulk_group_size
        if args.disable_pseudobulk:
            training.pseudobulk_group_size = 1
        if args.pseudobulk_pca_components is not None:
            training.pseudobulk_pca_components = args.pseudobulk_pca_components
        if args.resource_sample_seconds is not None:
            training.resource_sample_seconds = args.resource_sample_seconds
        if args.atac_layer:
            training.atac_layer = None if args.atac_layer == "none" else args.atac_layer
        training.device_preference = args.device
        if args.disable_feature_importance:
            training.enable_feature_importance = False
        if args.feature_importance_samples is not None:
            training.feature_importance_samples = args.feature_importance_samples
        if args.feature_importance_batch_size is not None:
            training.feature_importance_batch_size = args.feature_importance_batch_size
        if args.rf_n_estimators is not None:
            training.rf_n_estimators = args.rf_n_estimators
        if args.rf_max_depth is not None:
            training.rf_max_depth = args.rf_max_depth
        if args.rf_min_samples_leaf is not None:
            training.rf_min_samples_leaf = args.rf_min_samples_leaf
        if args.rf_max_features is not None:
            try:
                training.rf_max_features = float(args.rf_max_features)
            except ValueError:
                training.rf_max_features = args.rf_max_features
        if args.rf_bootstrap is not None:
            training.rf_bootstrap = args.rf_bootstrap.lower() == "true"
        training.validate()

        models = ModelConfig()
        if args.models:
            models.model_names = args.models
        if args.extra_models:
            models.extra_models = args.extra_models

        if args.multi_output and args.per_gene:
            parser.error("Cannot set both --multi-output and --per-gene")

        multi_output_mode = True
        if args.per_gene:
            multi_output_mode = False
        elif args.multi_output:
            multi_output_mode = True

        gene_list: Optional[list[str]] = list(args.genes) if args.genes else None
        if args.gene_manifest:
            manifest_path = Path(args.gene_manifest).expanduser().resolve()
            if not manifest_path.exists():
                parser.error(f"Gene manifest not found at {manifest_path}")
            manifest_genes = _load_manifest_genes(manifest_path)
            if not manifest_genes:
                parser.error(f"Gene manifest {manifest_path} did not contain any gene entries")
            gene_list = manifest_genes

        config = PipelineConfig(
            paths=paths,
            training=training,
            models=models,
            genes=gene_list,
            chromosomes=list(args.chromosomes) if args.chromosomes else None,
            max_genes=args.max_genes,
            chunk_total=args.chunk_total,
            chunk_index=args.chunk_index,
            multi_output=multi_output_mode,
        )

    run_name = args.run_name or f"spear_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config.run_name = run_name
    config.ensure_directories()

    log_path = configure_logging(config.paths.logs_dir, run_name)
    logger = get_logger(__name__)
    logger.info("Logging to %s", log_path)

    try:
        output_dir = run_pipeline(config)
    except Exception:
        logger.exception("Pipeline terminated with an error")
        logger.error("RUN_COMPLETE_STATUS=FAILURE")
        raise SystemExit(1)

    logger.info("Pipeline complete. Results stored in %s", output_dir)
    logger.info("RUN_COMPLETE_STATUS=SUCCESS")


def _config_from_json(payload: dict) -> PipelineConfig:
    base_dir = payload.get("base_dir", str(Path.cwd()))
    paths = PathsConfig.from_base(base_dir)
    training_payload = payload.get("training", {})
    models_payload = payload.get("models", {})

    training = TrainingConfig(**training_payload)
    training.validate()
    models = ModelConfig(**models_payload)

    return PipelineConfig(
        paths=paths,
        training=training,
        models=models,
        genes=payload.get("genes"),
        chromosomes=payload.get("chromosomes"),
        max_genes=payload.get("max_genes"),
        chunk_total=payload.get("chunk_total", 1),
        chunk_index=payload.get("chunk_index", 0),
        # Default to multi-output unless explicitly disabled in JSON payload
        multi_output=payload.get("multi_output", True),
    )


def _load_manifest_genes(manifest_path: Path) -> list[str]:
    text = manifest_path.read_text().splitlines()
    stripped = [line.strip() for line in text if line.strip() and not line.strip().startswith("#")]
    if not stripped:
        return []

    sniff_sample = "\n".join(stripped[:5])
    if any(delim in sniff_sample for delim in (",", "\t", ";")):
        try:
            dialect = csv.Sniffer().sniff(sniff_sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.get_dialect("excel")
        genes: list[str] = []
        with manifest_path.open("r", newline="") as handle:
            reader = csv.reader(handle, dialect)
            rows = [row for row in reader if row]
        if not rows:
            return []
        header_candidates = {value.strip().lower(): idx for idx, value in enumerate(rows[0])}
        gene_col = None
        for key in ("gene_name", "gene", "gene_id", "geneid"):
            if key in header_candidates:
                gene_col = header_candidates[key]
                break
        start_idx = 1 if gene_col is not None else 0
        if gene_col is None:
            gene_col = 0
        for row in rows[start_idx:]:
            if gene_col < len(row):
                value = row[gene_col].strip()
                if value:
                    genes.append(value)
        unique_ordered = list(dict.fromkeys(genes))
        return unique_ordered

    return list(dict.fromkeys(stripped))


if __name__ == "__main__":
    main()
