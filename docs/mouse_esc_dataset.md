# Mouse Embryonic Stem Cell (mESC) Multi-omics Dataset

## Overview

- Single-cell paired RNA-seq and ATAC-seq profiling of mouse embryonic stem cell differentiation.
- Captures developmental time points spanning E7.5 through E8.75 with CRISPR perturbation controls.
- Organism & strain: C57BL/6Babr mice.

## Source References

- Primary study: _Decoding gene regulation in the mouse embryo using single-cell multi-omics_ ([bioRxiv, 2022](https://www.biorxiv.org/content/10.1101/2022.06.15.496239v2)).
- GEO accession: [GSE205117](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205117).

## Sample Inventory

| Label            | GEO sample              | Modalities                                        |
| ---------------- | ----------------------- | ------------------------------------------------- |
| E7.5_rep1        | GSM6205416 / GSM6205427 | GEX (barcodes, features, matrix) + ATAC fragments |
| E7.5_rep2        | GSM6205417 / GSM6205428 | GEX + ATAC fragments                              |
| E7.75_rep1       | GSM6205418 / GSM6205429 | GEX + ATAC fragments                              |
| E8.0_rep1        | GSM6205419 / GSM6205430 | GEX + ATAC fragments                              |
| E8.0_rep2        | GSM6205420 / GSM6205431 | GEX + ATAC fragments                              |
| E8.5_CRISPR_T_KO | GSM6205421 / GSM6205432 | GEX + ATAC fragments                              |
| E8.5_CRISPR_T_WT | GSM6205422 / GSM6205433 | GEX + ATAC fragments                              |
| E8.5_rep1        | GSM6205423 / GSM6205434 | GEX + ATAC fragments                              |
| E8.5_rep2        | GSM6205424 / GSM6205435 | GEX + ATAC fragments                              |
| E8.75_rep1       | GSM6205425 / GSM6205436 | GEX + ATAC fragments                              |
| E8.75_rep2       | GSM6205426 / GSM6205437 | GEX + ATAC fragments                              |

## Data Access

The raw and processed data files are available from GEO accession [GSE205117](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205117). Download the required files for your analysis:

- **GEX matrices**: Per-sample barcodes, features, and count matrices (`.tsv.gz` or `.mtx.gz`)
- **ATAC fragments**: Per-sample fragment files for chromatin accessibility

Store downloaded files in appropriate directories under `data/embryonic/raw/` or `data/endothelial/raw/` depending on your dataset.
