# SPEAR Package (Single-cell Prediction of gene Expression from ATAC-seq Regression)

Python package housing the reusable components that power the SPEAR workflow:

- `config.py` – dataclasses describing filesystem layout, training hyperparameters, and model selections.
- `cli.py` – entrypoint for command-line execution (`spear` or `python -m spear.cli`).
- `data.py`, `training.py`, `evaluation.py`, `metrics.py` – data handling, model training loops, and evaluation logic.
- `visualization.py` – plotting utilities for diagnostic figures.

Install in editable mode for development:

```bash
pip install -e .
```

Run `spear --help` (or `python -m spear.cli --help`) for the full list of pipeline options.
