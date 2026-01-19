# Mouse Embryonic Multiome Dataset (GSE205117)

## Overview

- Mouse early organogenesis profiled with the 10x Genomics Multiome assay.
- Paired snRNA-seq and snATAC-seq across E7.5 to E8.75, including wild-type replicates and a Brachyury CRISPR T knockout condition.
- Organism & strain: C57BL/6Babr mice.

## Source References

- Primary study: _Decoding gene regulation in the mouse embryo using single-cell multi-omics_ ([bioRxiv, 2022](https://www.biorxiv.org/content/10.1101/2022.06.15.496239v2)).
- GEO accession: [GSE205117](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205117).

## Sample Inventory

The GEO series reports 11 samples spanning E7.5 to E8.75, including wild-type replicates and a Brachyury perturbation condition (CRISPR T knockout).

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

## Data Snapshot

- **ATAC:** 54,301 cells × 192,248 peaks (sparse float32 counts 1–4). Single `mESC` label with sample sizes ranging ~1.8k–10.7k cells.
- **RNA:** 54,301 cells × 32,285 genes (sparse integer counts up to 7,858 UMIs). QC metrics stored in `obs` (total counts, mitochondrial fraction, etc.).
- **Library sizes:** ATAC median 25k peaks/cell (IQR 14k–39k); RNA median 11k UMIs/cell (IQR 7k–17k).
- **Metadata:** Peak identifiers encode genomic coordinates; GTF parsing ensures gene IDs (e.g., `Kmt5b`) resolve consistently across modalities.

## CRISPR KO Replicates

- Do **not** merge CRISPR knockout replicates when preparing AnnData inputs. Each replicate should remain a distinct sample in `obs` (e.g., `obs["replicate"]`) so biological variance is preserved.
- When running pseudobulk or cross-validation, keep replicate identifiers intact; the pipeline assumes replicates were not combined upstream and will treat each as an independent group.

## Data Access

The raw and processed data files are available from GEO accession [GSE205117](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205117). Download the required files for your analysis:

- **GEX matrices**: Per-sample barcodes, features, and count matrices (`.tsv.gz` or `.mtx.gz`)
- **ATAC fragments**: Per-sample fragment files for chromatin accessibility

Store downloaded files under `data/embryonic/raw/`.
