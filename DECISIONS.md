# Decisions

## 1. Part 2 area
Chosen area: **Machine Learning**.

Reason: the supplied baseline already provides a deterministic anomaly-ranking benchmark, so an ML risk-ranking approach can be compared against the same operational task without changing the submission interface.

## 2. Target / label
The hidden scorer is not available during development. Field visits are therefore used only as an **operational proxy label** for development: a visit occurring in the seven days after a historical scoring week.

We do not claim this proxy is the challenge ground truth. Visit records can reflect scheduling, access problems, routine work, or other operational causes.

## 3. Leakage control
For every training week, features use telemetry strictly before that week's start. The final scored weeks are not used to create training labels.

## 4. Features
The model combines recent 7-day and trailing 28-day telemetry signals, meter-read success, and stable gateway metadata. Missing numeric values are retained as NaN because the chosen estimator can handle missing numeric values.

## 5. Ranking
Predicted risk is sorted descending. If scores tie, the normalized gateway ID is used as the deterministic secondary key. Exactly 15 unique gateways are emitted per week.

## 6. Why not use visit outcome as ground truth?
The challenge instructions explicitly distinguish the hidden scorer from field visits/outcomes and the engineer category. Treating visits as ground truth would overstate what the local data can establish. They are used only as a clearly named proxy for development.

## 7. Why not optimize for accuracy/F1?
The challenge evaluates the operational cost of the selected gateways. Accuracy/F1 can be misleading when the action is a top-15 ranking. The final external score should therefore be treated as authoritative; local proxy metrics are secondary diagnostics.

## 8. Partial telemetry
The implementation does not silently assume a gateway is healthy when telemetry is missing. Aggregations produce missing features, which the model handles. A future iteration should add explicit coverage features and sensitivity checks before final presentation.

## 9. Rejected alternatives
- **Random tie-breaking:** rejected because repeated runs could produce different submissions.
- **Using future telemetry:** rejected because it would leak information unavailable at decision time.
- **Using field-visit outcome as the hidden scorer:** rejected because the challenge materials explicitly distinguish those sources.
- **Deploying a hosted model/API:** rejected because runtime must work locally without a paid account or network dependency.
