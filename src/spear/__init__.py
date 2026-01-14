"""Gene regulatory network regression pipeline."""

from .config import PipelineConfig, PathsConfig, TrainingConfig, ModelConfig


def main(*args, **kwargs):  # pragma: no cover - thin wrapper for CLI entrypoint
    # Lazy import avoids double-import warnings when running `python -m spear.cli`.
    from .cli import main as _cli_main

    return _cli_main(*args, **kwargs)


__all__ = [
    "PipelineConfig",
    "PathsConfig",
    "TrainingConfig",
    "ModelConfig",
    "main",
]
