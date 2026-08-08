"""
scripts/find_best_run.py — Find the Best MLflow Run (Sessions 2 & 4)
======================================================================
USAGE:
    python scripts/find_best_run.py
    python scripts/find_best_run.py --metric accuracy --experiment churn_prediction

WHAT THIS DOES:
    Searches all runs in an MLflow experiment, sorts by the given metric
    (default: auc_roc), and prints the best Run ID — ready to copy-paste
    into `python src/evaluate.py --run_id ...` or
    `python scripts/register_model.py --run_id ...`
"""

import argparse

import mlflow


def find_best_run(experiment_name: str, metric: str) -> str:
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} DESC"],
    )

    if runs.empty:
        print(f"[find_best_run] No runs found for experiment '{experiment_name}'.")
        print("[find_best_run] Run `python src/train.py` first.")
        return None

    best = runs.iloc[0]
    run_id = best["run_id"]
    metric_value = best.get(f"metrics.{metric}", "N/A")

    print("\n" + "═" * 55)
    print(f"  BEST RUN — sorted by {metric}")
    print("═" * 55)
    print(f"  Run ID     : {run_id}")
    print(f"  {metric:<10} : {metric_value}")
    print("─" * 55)

    # Print all logged metrics for context
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    for col in metric_cols:
        name = col.replace("metrics.", "")
        print(f"  {name:<10} : {best[col]}")
    print("═" * 55)
    print(f"\n  Copy this Run ID:\n  >>> {run_id}\n")

    return run_id


def parse_args():
    parser = argparse.ArgumentParser(description="Find the best MLflow run by a given metric")
    parser.add_argument("--experiment", default="churn_prediction")
    parser.add_argument("--metric", default="auc_roc", choices=["accuracy", "auc_roc", "f1_score"])
    parser.add_argument("--tracking_uri", default="http://localhost:5001")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    find_best_run(args.experiment, args.metric)
