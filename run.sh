#!/usr/bin/env bash
set -euo pipefail
python -m src.ml_rank --data data --out results
python validate_submission.py results/predictions.csv
