"""Merge per-sample filtered AnnData objects into combined RNA and ATAC files.

This script scans the sibling `filtered/` directory (relative to this file) for
subdirectories containing `<sample>_RNA_qc.h5ad` and `<sample>_ATAC_qc.h5ad`.
All RNA objects are concatenated into a single AnnData and written to
`combined_RNA_qc.h5ad`; likewise for the ATAC objects.

Usage:
    python combine_filtered.py

The script attempts to be conservative about schema mismatches by aligning
features via an outer join and filling missing entries with zeros.
"""


from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import scipy.sparse as sp
import pandas as pd


HERE = Path(__file__).resolve().parent
FILTERED_DIR = HERE.parent / "filtered"
OUTPUT_RNA = FILTERED_DIR / "combined_RNA_qc.h5ad"
OUTPUT_ATAC = FILTERED_DIR / "combined_ATAC_qc.h5ad"


def collect_sample_dirs(filtered_dir: Path) -> List[Path]:
    """Return sorted list of sample directories containing QC'd AnnData files."""
    sample_dirs: List[Path] = [p for p in filtered_dir.iterdir() if p.is_dir()]
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found in {filtered_dir}")
    return sorted(sample_dirs)


def load_anndata(sample_dir: Path, suffix: str) -> ad.AnnData:
    """Load an AnnData object (`suffix` should be either 'RNA_qc' or 'ATAC_qc')."""
    sample_name = sample_dir.name
    file_path = sample_dir / f"{sample_name}_{suffix}.h5ad"
    if not file_path.exists():
        raise FileNotFoundError(f"Expected file {file_path} is missing")

    adata = ad.read_h5ad(file_path)

    # Ensure convenient metadata columns are present
    if "barcode" not in adata.obs:
        adata.obs["barcode"] = adata.obs_names
    adata.obs["sample"] = sample_name
    adata.obs["sample"] = adata.obs["sample"].astype("category")

    # Guard against duplicated feature names
    adata.var_names_make_unique()

    # Normalize storage: CSR + float32 saves memory when concatenating
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr()
    else:
        adata.X = sp.csr_matrix(adata.X)
    if adata.X.dtype != np.float32:
        adata.X = adata.X.astype(np.float32)
    return adata


def assemble_modality(sample_dirs: List[Path], suffix: str) -> Tuple[ad.AnnData, List[str]]:
    matrix: sp.csr_matrix | None = None
    obs_frames: List[pd.DataFrame] = []
    var_df: pd.DataFrame | None = None
    sample_names: List[str] = []

    for idx, sample_dir in enumerate(sample_dirs):
        adata = load_anndata(sample_dir, suffix)
        sample_names.append(sample_dir.name)

        if var_df is None:
            var_df = adata.var.copy()
        else:
            if not adata.var_names.equals(var_df.index):
                raise ValueError(
                    f"Feature order mismatch for {suffix} between {sample_dir.name} and first sample"
                )

        obs_frames.append(adata.obs.copy())

        matrix = adata.X if matrix is None else sp.vstack([matrix, adata.X], format="csr")

    if var_df is None or matrix is None:
        raise ValueError(f"No data assembled for suffix {suffix}")

    combined_obs = pd.concat(obs_frames, axis=0)
    combined = ad.AnnData(X=matrix, obs=combined_obs, var=var_df.copy())
    combined.var_names = var_df.index
    combined.obs_names = combined_obs.index
    combined.obs["sample"] = combined.obs["sample"].astype("category")
    return combined, sample_names


def merge_modalities() -> Dict[str, ad.AnnData]:
    sample_dirs = collect_sample_dirs(FILTERED_DIR)
    combined_rna, sample_names = assemble_modality(sample_dirs, "RNA_qc")
    combined_atac, _ = assemble_modality(sample_dirs, "ATAC_qc")
    return {"rna": combined_rna, "atac": combined_atac, "samples": sample_names}


def main() -> None:
    combined = merge_modalities()
    combined["rna"].write_h5ad(OUTPUT_RNA)
    combined["atac"].write_h5ad(OUTPUT_ATAC)

    sample_count = len(combined["samples"])
    print(
        f"Wrote RNA: {OUTPUT_RNA} (cells={combined['rna'].n_obs}, genes={combined['rna'].n_vars}, "
        f"samples={sample_count})"
    )
    print(
        f"Wrote ATAC: {OUTPUT_ATAC} (cells={combined['atac'].n_obs}, peaks={combined['atac'].n_vars}, "
        f"samples={sample_count})"
    )


if __name__ == "__main__":
    main()
