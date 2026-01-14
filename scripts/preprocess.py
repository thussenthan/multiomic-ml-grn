"""QC and basic preprocessing for paired scRNA-seq and scATAC-seq data.

This module is still primarily a filtering/prefiltering workflow that trims out
low-quality cells/features before any heavy transformations downstream.

Current workflow per sample:
1. Load raw 10x RNA matrix (features + matrix + barcodes) and a large shared ATAC peak matrix.
2. Harmonize / normalize barcodes between RNA and ATAC modalities (keep intersection).
3. Perform light RNA QC (mitochondrial content filter) and cache AnnData objects.
4. Apply simple filtering thresholds (min genes / min cells) for RNA and ATAC.
5. Write out the ordered list of retained peaks ("Peaks.txt").

Current outputs written per sample directory:
    - <sample>_RNA_qc.h5ad / <sample>_ATAC_qc.h5ad : cached QC-filtered AnnData objects
    - Peaks.txt : ordered list of retained peak identifiers (one per line)

Assumptions:
    - Access to a shared scATAC peak matrix file (tab-delimited) whose columns include all sample barcodes.
    - Each sample has a 10x-formatted RNA directory with a single features.tsv.gz defining the prefix.
"""

import pandas as pd
import scanpy as sc
import os
import warnings
import scipy.sparse as sp
import logging
from pathlib import Path
from anndata import AnnData
import multiprocessing as mp

warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group`")


def load_rna_adata(sample_raw_data_dir: str) -> sc.AnnData:
    """Load a 10x-formatted RNA directory into an AnnData object.

    The function infers the file prefix by locating the single features.tsv.gz file,
    then delegates to scanpy's read_10x_mtx with that prefix.
    """
    # Detect the (single) features.tsv.gz file to infer the 10x file prefix
    features = [f for f in os.listdir(sample_raw_data_dir) if f.endswith("features.tsv.gz")]
    assert len(features) == 1, f"Expected 1 features.tsv.gz, found {features}"

    prefix = features[0].replace("features.tsv.gz", "")
    logging.info(f"Detected RNA prefix: {prefix}")

    adata = sc.read_10x_mtx(
        sample_raw_data_dir,
        var_names="gene_symbols",
        make_unique=True,
        prefix=prefix
    )
    return adata

def get_adata_from_peakmatrix(peak_matrix_file: str, label: pd.DataFrame, sample_name: str) -> AnnData:
    """Load selected columns from a large peak matrix into an AnnData object.

    The peak matrix is assumed to be a tab-separated text file with first column = peak IDs
    and subsequent columns = cell barcodes. To save memory/time, only barcodes present in
    the provided label DataFrame are retained.
    """
    # Read a small number of rows just to capture the header (column names) quickly
    all_cols = pd.read_csv(peak_matrix_file, sep="\t", nrows=10).columns
    
    # Determine overlap between available ATAC barcodes and RNA barcodes
    matching_barcodes = set(label["barcode_use"]) & set(all_cols)

    # Map original column index -> barcode for selective reading
    col_map = {i: bc for i, bc in enumerate(all_cols)}

    # Always include first column (peak IDs) plus those whose barcode we want
    keep_indices = [0] + [i for i, bc in col_map.items() if bc in matching_barcodes]

    # Read only needed columns (reduces memory footprint substantially)
    peak_matrix = pd.read_csv(
        peak_matrix_file,
        sep="\t",
        usecols=keep_indices,
        index_col=0
    )

    # Normalize column names to the selected barcodes in order
    new_cols = [col_map[i] for i in keep_indices[1:]]
    peak_matrix.columns = new_cols

    # Build AnnData (cells x peaks) -> transpose of peak_matrix values
    X = sp.csr_matrix(peak_matrix.values)
    adata_ATAC = AnnData(X=X.T)

    # Cell metadata
    adata_ATAC.obs_names = new_cols
    adata_ATAC.obs["barcode"] = new_cols
    adata_ATAC.obs["sample"] = sample_name
    adata_ATAC.obs["label"] = label.set_index("barcode_use").loc[new_cols, "label"].values

    # Peak metadata
    adata_ATAC.var_names = peak_matrix.index
    adata_ATAC.var["gene_ids"] = peak_matrix.index  # Re-using key name 'gene_ids' for peaks (downstream expectation?)

    return adata_ATAC

def process_sample(sample_name: str):
    """Process a single sample end-to-end and write outputs to disk.

    Caches intermediate QC AnnData files so re-runs skip raw loading / QC.
    """
    sample_data_dir = os.path.join(SAMPLE_INPUT_DIR, sample_name)
    os.makedirs(sample_data_dir, exist_ok=True)

    # If QC'd AnnData files already exist, reuse them (speeds up iterative work)
    if os.path.exists(os.path.join(sample_data_dir, f"{sample_name}_RNA_qc.h5ad")) \
       and os.path.exists(os.path.join(sample_data_dir, f"{sample_name}_ATAC_qc.h5ad")):
        adata_RNA = sc.read_h5ad(os.path.join(sample_data_dir, f"{sample_name}_RNA_qc.h5ad"))
        adata_ATAC = sc.read_h5ad(os.path.join(sample_data_dir, f"{sample_name}_ATAC_qc.h5ad"))
    else:
        # --- Load raw RNA data (10x files) ---
        sample_raw_data_dir = os.path.join(RAW_MESC_DATA_DIR, sample_name)
        adata_RNA = load_rna_adata(sample_raw_data_dir)
        # Normalize barcode formatting and prepend sample name (avoid collisions across samples)
        adata_RNA.obs_names = [(sample_name + "." + i).replace("-", ".") for i in adata_RNA.obs_names]
        logging.info(f"[{sample_name}] Found {len(adata_RNA.obs_names)} RNA barcodes")

        # Build simple label DataFrame (all cells same label here: mESC)
        label = pd.DataFrame({"barcode_use": adata_RNA.obs_names,
                              "label": ["mESC"] * len(adata_RNA.obs_names)})

        # Load ATAC counts (subset to barcodes present in RNA)
        adata_ATAC = get_adata_from_peakmatrix(MESC_PEAK_MATRIX_FILE, label, sample_name)

        # Synchronize barcodes across modalities (keep only intersection)
        adata_RNA.obs['barcode'] = adata_RNA.obs_names
        common_barcodes = adata_RNA.obs['barcode'].isin(adata_ATAC.obs['barcode'])
        adata_RNA = adata_RNA[common_barcodes].copy()
        adata_ATAC = adata_ATAC[adata_ATAC.obs['barcode'].isin(adata_RNA.obs['barcode'])].copy()

        # --- RNA QC: mitochondrial fraction filter ---
        adata_RNA.var['mt'] = adata_RNA.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata_RNA, qc_vars=["mt"], inplace=True)
        adata_RNA = adata_RNA[adata_RNA.obs.pct_counts_mt < 5].copy()
        adata_RNA.var_names_make_unique()
        adata_RNA.var['gene_ids'] = adata_RNA.var.index

        # Cache QC objects
        adata_RNA.write_h5ad(os.path.join(sample_data_dir, f"{sample_name}_RNA_qc.h5ad"))
        adata_ATAC.write_h5ad(os.path.join(sample_data_dir, f"{sample_name}_ATAC_qc.h5ad"))

    # --- Simple filtering (remove very low complexity cells / features) ---
    sc.pp.filter_cells(adata_RNA, min_genes=200)
    sc.pp.filter_genes(adata_RNA, min_cells=3)
    sc.pp.filter_cells(adata_ATAC, min_genes=200)
    sc.pp.filter_genes(adata_ATAC, min_cells=3)
    
    pd.DataFrame(adata_ATAC.var['gene_ids']).to_csv(
        os.path.join(sample_data_dir, "Peaks.txt"), header=None, index=None
    )

    logging.info(f"[{sample_name}] Finished processing")
    return sample_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = PROJECT_ROOT / "filtered"  # Root project directory
RAW_MESC_DATA_DIR = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS/DS014_DOI496239_MOUSE_ESC_RAW_FILES"  # Location of per-sample 10x RNA dirs
MESC_PEAK_MATRIX_FILE = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS/DS014_DOI496239_MOUSE_ESCDAYS7AND8/scATAC_PeakMatrix.txt"  # Large shared scATAC peak matrix

# MM10_GENOME_DIR = os.path.join(PROJECT_DIR, "")
# MM10_GENE_TSS_FILE = os.path.join(PROJECT_DIR, "")
SAMPLE_INPUT_DIR = os.fspath(PROJECT_DIR)  # Where per-sample outputs are written (currently project root)
# OUTPUT_DIR = os.path.join(PROJECT_DIR, "")

def main():
    # List of sample directory names to process. (Note: 'E7.5_rep1' appears twice; verify if intentional)
    sample_name_list = ["E7.5_rep1", "E7.5_rep1", "E7.75_rep1", "E8.0_rep2", "E8.5_rep2",
                        "E8.75_rep2", "E7.5_rep2", "E8.0_rep1", "E8.5_rep1", "E8.75_rep1"]
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Parallelize per-sample work. Tune processes to available CPU cores / I/O constraints.
    with mp.Pool(processes=12) as pool:  # adjust #processes to #CPUs available
        results = pool.map(process_sample, sample_name_list)

    logging.info(f"Completed samples: {results}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
