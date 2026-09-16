from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

SCORED_WEEKS = pd.date_range("2026-02-02", "2026-03-23", freq="7D")
FEATURES = [
    "offline_hours_7d", "offline_hours_28d", "disconnections_7d", "disconnections_28d",
    "reboots_7d", "reboots_28d", "reboot_duration_7d", "rssi_bad_share_7d",
    "crc_bad_rate_7d", "tx_busy_rate_7d", "load_mean_7d", "memfree_mean_7d",
    "uptime_mean_7d", "meter_success_rate", "age_days", "meters_installed",
]


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(":", "", regex=False).str.upper()


def load_telemetry(data_dir: Path) -> pd.DataFrame:
    files = sorted((data_dir / "telemetry").glob("month=*/part-*.parquet"))
    if not files:
        raise FileNotFoundError("No telemetry parquet files found under data/telemetry")
    cols = [
        "gateway_id", "ts_utc", "offline_duration_sec", "disconnection_cnt", "reboot_cnt",
        "reboot_duration_sec", "rssi_bad", "rx_crc_bad", "number_of_messages",
        "tx_busy", "avg_load1", "avg_memfree", "avg_uptime",
    ]
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{f}: missing telemetry columns: {missing}")
        frames.append(df[cols])
    out = pd.concat(frames, ignore_index=True)
    out["gateway_id"] = norm_id(out["gateway_id"])
    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True, errors="coerce")
    for c in cols[2:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["gateway_id", "ts_utc"])


def load_metadata(data_dir: Path) -> pd.DataFrame:
    master = pd.read_csv(data_dir / "gateway_master.csv", encoding="cp1252")
    master["gateway_id"] = norm_id(master["gateway_id"])
    master["installed_on"] = pd.to_datetime(master["installed_on"], errors="coerce")
    return master


def load_meter(data_dir: Path) -> pd.DataFrame:
    m = pd.read_csv(data_dir / "meter_read_success.csv")
    m["gateway_id"] = norm_id(m["gateway_id"])
    m["week_start"] = pd.to_datetime(m["week_start"], errors="coerce")
    m["meter_success_rate"] = np.divide(m["meters_read"], m["meters_expected"].replace(0, np.nan))
    return m[["gateway_id", "week_start", "meter_success_rate"]]


def aggregate_week(telemetry: pd.DataFrame, week_start: pd.Timestamp, window_days: int) -> pd.DataFrame:
    end = week_start
    start = week_start - pd.Timedelta(days=window_days)
    x = telemetry[(telemetry.ts_utc >= start.tz_localize("UTC")) & (telemetry.ts_utc < end.tz_localize("UTC"))].copy()
    if x.empty:
        return pd.DataFrame(columns=["gateway_id"] + FEATURES[:13])
    x["crc_bad_rate"] = x["rx_crc_bad"] / x["number_of_messages"].replace(0, np.nan)
    x["tx_busy_rate"] = x["tx_busy"] / x["number_of_messages"].replace(0, np.nan)
    x["rssi_bad_share"] = x["rssi_bad"].fillna(0) > 0
    g = x.groupby("gateway_id", observed=True)
        # Rename these when called for 28d; aggregation uses same names then caller suffixes.
    return g.agg(
        offline_hours=("offline_duration_sec", lambda s: s.sum(skipna=True) / 3600),
        disconnections=("disconnection_cnt", "sum"), reboots=("reboot_cnt", "sum"),
        reboot_duration=("reboot_duration_sec", "sum"), rssi_bad_share=("rssi_bad_share", "mean"),
        crc_bad_rate=("crc_bad_rate", "mean"), tx_busy_rate=("tx_busy_rate", "mean"),
        load_mean=("avg_load1", "mean"), memfree_mean=("avg_memfree", "mean"),
        uptime_mean=("avg_uptime", "mean"),
    ).reset_index()


def build_features(telemetry: pd.DataFrame, master: pd.DataFrame, meter: pd.DataFrame, weeks: Iterable[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for w in weeks:
        a = aggregate_week(telemetry, w, 7).rename(columns={
            "offline_hours":"offline_hours_7d", "disconnections":"disconnections_7d", "reboots":"reboots_7d",
            "reboot_duration":"reboot_duration_7d", "rssi_bad_share":"rssi_bad_share_7d",
            "crc_bad_rate":"crc_bad_rate_7d", "tx_busy_rate":"tx_busy_rate_7d", "load_mean":"load_mean_7d",
            "memfree_mean":"memfree_mean_7d", "uptime_mean":"uptime_mean_7d"})
        b = aggregate_week(telemetry, w, 28).rename(columns={
            "offline_hours":"offline_hours_28d", "disconnections":"disconnections_28d", "reboots":"reboots_28d"})
        b = b[["gateway_id","offline_hours_28d","disconnections_28d","reboots_28d"]]
        f = a.merge(b, on="gateway_id", how="outer")
        f["week_start"] = w
        rows.append(f)
    out = pd.concat(rows, ignore_index=True)
    out = out.merge(meter, on=["gateway_id","week_start"], how="left")
    meta = master[["gateway_id","installed_on","n_meters_installed"]].drop_duplicates("gateway_id")
    out = out.merge(meta, on="gateway_id", how="left")
    out["age_days"] = (out["week_start"] - out["installed_on"]).dt.days
    out["meters_installed"] = pd.to_numeric(out["n_meters_installed"], errors="coerce")
    out = out.drop(columns=["installed_on","n_meters_installed"], errors="ignore")
    out[FEATURES] = out[FEATURES].apply(pd.to_numeric, errors="coerce")
    return out


def make_proxy_labels(features: pd.DataFrame, visits: pd.DataFrame) -> pd.Series:
    """Operational proxy only: a visit requested/visited in the following 7 days.
    This is NOT the hidden challenge ground truth and is not used as the submission scorer.
    """
    v = visits.copy()
    v["gateway_id"] = norm_id(v["gateway_id"])
    v["visited_on"] = pd.to_datetime(v["visited_on"], errors="coerce")
    v = v.dropna(subset=["visited_on"])
    pairs = set(zip(v.gateway_id, v.visited_on.dt.normalize()))
    y = []
    for r in features.itertuples(index=False):
        start = pd.Timestamp(r.week_start)
        hit = any((r.gateway_id, (start + pd.Timedelta(days=d)).normalize()) in pairs for d in range(7))
        y.append(int(hit))
    return pd.Series(y, index=features.index, name="proxy_visit_next_7d")


def train_and_score(data_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    telemetry = load_telemetry(data_dir)
    master = load_metadata(data_dir)
    meter = load_meter(data_dir)
    visits = pd.read_csv(data_dir / "field_visits.csv")

    historical_weeks = pd.date_range("2025-09-01", "2026-01-26", freq="7D")
    # Include scored weeks for feature construction, but labels are never computed from future scored weeks.
    final_weeks = pd.date_range("2026-02-02", "2026-03-23", freq="7D")
    train = build_features(telemetry, master, meter, historical_weeks)
    y = make_proxy_labels(train, visits)

    # Class imbalance is expected. HistGradientBoosting handles missing numeric values.
    X = train[FEATURES].replace([np.inf, -np.inf], np.nan)
    model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=42)
    model.fit(X, y)

    score = build_features(telemetry, master, meter, final_weeks)
    Xs = score[FEATURES].replace([np.inf, -np.inf], np.nan)
    score["score"] = model.predict_proba(Xs)[:, 1]
    score["score"] = score["score"].fillna(0.0)

    # Deterministic tie-break: score desc, gateway id asc.
    predictions = []
    for w, g in score.groupby("week_start", sort=True):
        g = g.sort_values(["score", "gateway_id"], ascending=[False, True]).head(15).copy()
        g["rank"] = np.arange(1, len(g) + 1)
        g["reason"] = g.apply(lambda r: (
            f"ML risk={r.score:.4f}; recent offline={r.offline_hours_7d:.2f}h, "
            f"disconnects={r.disconnections_7d:.0f}, reboots={r.reboots_7d:.0f}."
        ), axis=1)
        predictions.append(g[["week_start","rank","gateway_id","score","reason"]])
    pred = pd.concat(predictions, ignore_index=True)
    pred["week_start"] = pd.to_datetime(pred.week_start).dt.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / "predictions.csv", index=False)

    model_blob = json.dumps({"features": FEATURES, "model": model.get_params()}, sort_keys=True).encode()
    meta = {"model_type": "HistGradientBoostingClassifier", "random_state": 42,
            "training_label": "field visit in following 7 days (proxy only)",
            "feature_count": len(FEATURES), "model_config_sha256": hashlib.sha256(model_blob).hexdigest()}
    (out_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return pred, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    pred, meta = train_and_score(Path(args.data), Path(args.out))
    print(f"Wrote {len(pred)} rows to {Path(args.out)/'predictions.csv'}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
