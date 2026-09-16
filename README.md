# LPDG Innovation Hub Selection Challenge 2026

## Chosen Part 2 area: Machine Learning

This repository implements the required gateway-ranking submission and an ML-based ranking approach.

### Required output

The submission contains exactly 15 gateway picks for each of the 8 scored weeks (120 rows total), with deterministic ranks 1–15.

### Run locally

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.ml_rank --data data --out results
python validate_submission.py results/predictions.csv
```

No network connection or paid API is required at runtime.

## Method

The ML model uses historical gateway-week features derived only from telemetry available before each week. Features include recent and trailing offline time, disconnections, reboots, reboot duration, radio-quality indicators, traffic/load indicators, meter-read success, installation age, and installed meter count.

For the final scored weeks, gateways are ranked by predicted operational risk. Ties are resolved deterministically by gateway ID.

The training label is an **operational proxy**: whether a field visit occurred in the following seven days. It is explicitly not treated as the hidden challenge ground truth. The hidden scorer remains the authoritative evaluation for the final submission.

## Validation

The model-selection plan is time-aware: training examples precede the final scoring period. Additional evaluation should report performance separately on later weeks and on gateways not seen during training. Because the challenge scorer is hidden, local validation is an imperfect proxy and its limitations are documented in `DECISIONS.md`.

## Files

- `baseline_3sigma.py` — supplied challenge baseline.
- `validate_submission.py` — supplied submission validator.
- `src/ml_rank.py` — feature generation, model training and ranking.
- `DECISIONS.md` — important methodological decisions and rejected alternatives.
- `AI-USAGE.md` — AI assistance disclosure.
- `results/predictions.csv` — generated submission file after running `run.sh`.
- ## 🎥 Project Demo

[▶️ Watch the Project Demonstration](https://youtu.be/N0G6gflJt_g)
