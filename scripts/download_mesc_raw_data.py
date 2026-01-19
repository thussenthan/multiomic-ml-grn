#!/usr/bin/env python3
"""Download raw MESC multi-omics files listed in the original Slurm script."""


import argparse
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_DOWNLOAD_DIR = Path.cwd() / "data" / "raw"

SAMPLES: dict[str, list[str]] = {
    "E7.5_rep1": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205416/"
        "suppl/GSM6205416%5FE7.5%5Frep1%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205416/"
        "suppl/GSM6205416%5FE7.5%5Frep1%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205416/"
        "suppl/GSM6205416%5FE7.5%5Frep1%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205427/"
        "suppl/GSM6205427%5FE7.5%5Frep1%5FATAC%5Ffragments.tsv.gz",
    ],
    "E7.5_rep2": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205417/"
        "suppl/GSM6205417%5FE7.5%5Frep2%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205417/"
        "suppl/GSM6205417%5FE7.5%5Frep2%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205417/"
        "suppl/GSM6205417%5FE7.5%5Frep2%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205428/"
        "suppl/GSM6205428%5FE7.5%5Frep2%5FATAC%5Ffragments.tsv.gz",
    ],
    "E7.75_rep1": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205418/"
        "suppl/GSM6205418%5FE7.75%5Frep1%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205418/"
        "suppl/GSM6205418%5FE7.75%5Frep1%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205418/"
        "suppl/GSM6205418%5FE7.75%5Frep1%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205429/"
        "suppl/GSM6205429%5FE7.75%5Frep1%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.0_rep1": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205419/"
        "suppl/GSM6205419%5FE8.0%5Frep1%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205419/"
        "suppl/GSM6205419%5FE8.0%5Frep1%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205419/"
        "suppl/GSM6205419%5FE8.0%5Frep1%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205430/"
        "suppl/GSM6205430%5FE8.0%5Frep1%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.0_rep2": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205420/"
        "suppl/GSM6205420%5FE8.0%5Frep2%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205420/"
        "suppl/GSM6205420%5FE8.0%5Frep2%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205420/"
        "suppl/GSM6205420%5FE8.0%5Frep2%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205431/"
        "suppl/GSM6205431%5FE8.0%5Frep2%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.5_CRISPR_T_KO": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205421/"
        "suppl/GSM6205421%5FE8.5%5FCRISPR%5FT%5FKO%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205421/"
        "suppl/GSM6205421%5FE8.5%5FCRISPR%5FT%5FKO%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205421/"
        "suppl/GSM6205421%5FE8.5%5FCRISPR%5FT%5FKO%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205432/"
        "suppl/GSM6205432%5FE8.5%5FCRISPR%5FT%5FKO%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.5_CRISPR_T_WT": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205422/"
        "suppl/GSM6205422%5FE8.5%5FCRISPR%5FT%5FWT%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205422/"
        "suppl/GSM6205422%5FE8.5%5FCRISPR%5FT%5FWT%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205422/"
        "suppl/GSM6205422%5FE8.5%5FCRISPR%5FT%5FWT%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205433/"
        "suppl/GSM6205433%5FE8.5%5FCRISPR%5FT%5FWT%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.5_rep1": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205423/"
        "suppl/GSM6205423%5FE8.5%5Frep1%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205423/"
        "suppl/GSM6205423%5FE8.5%5Frep1%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205423/"
        "suppl/GSM6205423%5FE8.5%5Frep1%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205434/"
        "suppl/GSM6205434%5FE8.5%5Frep1%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.5_rep2": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205424/"
        "suppl/GSM6205424%5FE8.5%5Frep2%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205424/"
        "suppl/GSM6205424%5FE8.5%5Frep2%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205424/"
        "suppl/GSM6205424%5FE8.5%5Frep2%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205435/"
        "suppl/GSM6205435%5FE8.5%5Frep2%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.75_rep1": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205425/"
        "suppl/GSM6205425%5FE8.75%5Frep1%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205425/"
        "suppl/GSM6205425%5FE8.75%5Frep1%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205425/"
        "suppl/GSM6205425%5FE8.75%5Frep1%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205436/"
        "suppl/GSM6205436%5FE8.75%5Frep1%5FATAC%5Ffragments.tsv.gz",
    ],
    "E8.75_rep2": [
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205426/"
        "suppl/GSM6205426%5FE8.75%5Frep2%5FGEX%5Fbarcodes.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205426/"
        "suppl/GSM6205426%5FE8.75%5Frep2%5FGEX%5Ffeatures.tsv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205426/"
        "suppl/GSM6205426%5FE8.75%5Frep2%5FGEX%5Fmatrix.mtx.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6205nnn/GSM6205437/"
        "suppl/GSM6205437%5FE8.75%5Frep2%5FATAC%5Ffragments.tsv.gz",
    ],
}


def download_file(url: str, destination: Path, retries: int = 5, delay_sec: int = 5) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".partial")
    tmp_path.unlink(missing_ok=True)

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            tmp_path.replace(destination)
            return
        except urllib.error.URLError as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            time.sleep(delay_sec)


def sanitize_filename(url: str) -> str:
    encoded = url.rsplit("/", 1)[-1]
    decoded = urllib.parse.unquote(encoded)
    return decoded.replace("%5F", "_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download MESC raw multi-omic files.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help=f"Destination directory (default: {DEFAULT_DOWNLOAD_DIR})",
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for sample_label, urls in sorted(SAMPLES.items()):
        sample_dir = output_root / sample_label
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {sample_label} -> {sample_dir}")

        for url in urls:
            if not url:
                continue
            filename = sanitize_filename(url)
            dest_path = sample_dir / filename
            if dest_path.exists():
                print(f"  - {filename} exists, skipping")
                continue
            print(f"  - Fetching {filename}")
            download_file(url, dest_path)
        print()

    print("Download complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
