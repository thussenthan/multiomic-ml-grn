
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PathsConfig:
    base_dir: Path
    atac_path: Path
    rna_path: Path
    gtf_path: Path
    output_dir: Path
    logs_dir: Path
    figures_dir: Path

    @classmethod
    def from_base(
        cls,
        base_dir: str | Path,
        atac_filename: str = "combined_ATAC_qc.h5ad",
        rna_filename: str = "combined_RNA_qc.h5ad",
        gtf_filename: str = "GCF_000001635.27_genomic.gtf",
    ) -> "PathsConfig":
        root = Path(base_dir).expanduser().resolve()

        def _resolve(filename: str, fallback_dirs: list[str]) -> Path:
            candidate = (root / filename).expanduser()
            if candidate.exists():
                return candidate.resolve()
            for rel_dir in fallback_dirs:
                alt = (root / rel_dir / filename).expanduser()
                if alt.exists():
                    return alt.resolve()
            searched = [str((root / filename).expanduser())] + [
                str((root / rel_dir / filename).expanduser()) for rel_dir in fallback_dirs
            ]
            raise FileNotFoundError(
                f"Could not locate '{filename}'. Looked in: {', '.join(searched)}"
            )

        fallback_data_dirs = ["data/embryonic/processed", "data/raw"]
        atac_path = _resolve(atac_filename, fallback_data_dirs)
        rna_path = _resolve(rna_filename, fallback_data_dirs)
        gtf_path = _resolve(gtf_filename, ["data/reference"])
        output_root = (root / "output").resolve()
        output_dir = (output_root / "results").resolve()
        logs_dir = (output_root / "logs").resolve()
        figures_dir = (root / "analysis" / "figs").resolve()
        return cls(root, atac_path, rna_path, gtf_path, output_dir, logs_dir, figures_dir)


@dataclass
class TrainingConfig:
    window_bp: int = 10_000
    bin_size_bp: int = 500
    k_folds: int = 5
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    random_state: int = 42
    device_preference: str = "cuda"
    scaler: Optional[str] = "standard"
    min_cells_per_gene: int = 100
    min_expression: float = 0.0
    log1p_transform: bool = False
    target_scaler: Optional[str] = "standard"
    force_target_scaling: bool = False
    enable_smoothing: bool = True
    smoothing_k: int = 20
    smoothing_pca_components: int = 10
    pseudobulk_group_size: int = 1
    pseudobulk_pca_components: int = 10

    min_expression_fraction: float = 0.10
    rf_n_estimators: Optional[int] = None
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: Optional[int] = None
    rf_max_features: float | str | None = None
    rf_bootstrap: Optional[bool] = None
    track_history: bool = True
    history_metrics: List[str] = field(default_factory=lambda: ["mse", "pearson", "spearman"])
    group_key: Optional[str] = "sample"
    atac_layer: Optional[str] = "tfidf"
    rna_expression_layer: Optional[str] = "log1p_cpm"
    resource_sample_seconds: float = 60.0

    def validate(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Train/val/test fractions must sum to 1.0")
        if self.k_folds < 2:
            raise ValueError("k_folds must be at least 2")
        if self.window_bp <= 0 or self.bin_size_bp <= 0:
            raise ValueError("window_bp and bin_size_bp must be positive")
        if self.smoothing_k < 1:
            raise ValueError("smoothing_k must be >= 1")
        if self.smoothing_pca_components < 1:
            raise ValueError("smoothing_pca_components must be >= 1")
        if self.pseudobulk_group_size < 1:
            raise ValueError("pseudobulk_group_size must be >= 1")
        if self.pseudobulk_pca_components < 1:
            raise ValueError("pseudobulk_pca_components must be >= 1")
        if not (0.0 <= self.min_expression_fraction <= 1.0):
            raise ValueError("min_expression_fraction must be within [0, 1]")
        if self.rf_n_estimators is not None and self.rf_n_estimators <= 0:
            raise ValueError("rf_n_estimators must be positive when specified")
        if self.rf_max_depth is not None and self.rf_max_depth <= 0:
            raise ValueError("rf_max_depth must be positive when specified")
        if self.rf_min_samples_leaf is not None and self.rf_min_samples_leaf <= 0:
            raise ValueError("rf_min_samples_leaf must be positive when specified")
        if isinstance(self.rf_max_features, float) and not (0.0 < self.rf_max_features <= 1.0):
            raise ValueError("rf_max_features as a float must be within (0, 1]")
        if self.k_folds > 1 and self.group_key is not None and not self.group_key:
            raise ValueError("group_key must be a non-empty string when provided")
        if self.resource_sample_seconds <= 0:
            raise ValueError("resource_sample_seconds must be positive")


@dataclass
class ModelConfig:
    model_names: List[str] = field(
        default_factory=lambda: [
            "cnn",
            "rnn",
            "lstm",
            "mlp",
            "xgboost",
            "random_forest",
        ]
    )
    extra_models: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    paths: PathsConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    genes: Optional[List[str]] = None
    chromosomes: Optional[List[str]] = None
    max_genes: Optional[int] = None
    chunk_total: int = 1
    chunk_index: int = 0
    # Default to cell-wise multi-output unless explicitly turned off
    multi_output: bool = True
    run_name: Optional[str] = None


    def ensure_directories(self) -> None:
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.figures_dir.mkdir(parents=True, exist_ok=True)

    def all_models(self) -> List[str]:
        return list(dict.fromkeys(self.models.model_names + self.models.extra_models))
