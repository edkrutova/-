#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- IO / parsing -----------------------------

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_run_components(report_path: Path, runs_root: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract (dataset, vtype, classifier, run_id) from report path."""
    try:
        rel = report_path.relative_to(runs_root)
    except Exception:
        return None, None, None, None

    parts = rel.parts
    # runs/<dataset>/<vtype>/<cls>/<run_id>/report-*.json  => at least 5 parts
    if len(parts) < 5:
        return None, None, None, None

    return parts[0], parts[1], parts[2], parts[3]


def _cm_to_tnfpfn_tp(cm: Any):
    """Return (tn, fp, fn, tp) if cm is 2x2; else (None,...)."""
    if cm is None:
        return None, None, None, None
    try:
        arr = np.asarray(cm)
        if arr.shape == (2, 2):
            return int(arr[0, 0]), int(arr[0, 1]), int(arr[1, 0]), int(arr[1, 1])
    except Exception:
        pass
    return None, None, None, None


def load_reports(runs_root: Path) -> pd.DataFrame:
    rows = []
    for rp in sorted(runs_root.rglob("report-*.json")):
        data = _read_json(rp)
        if not isinstance(data, dict):
            continue

        dataset, vtype, cls, run_id = _parse_run_components(rp, runs_root)
        metrics = data.get("metrics", {}) if isinstance(data.get("metrics", {}), dict) else {}
        times = data.get("times", {}) if isinstance(data.get("times", {}), dict) else {}
        vec_params = data.get("vec_params", {})
        cls_params = data.get("cls_params", {})

        tn, fp, fn, tp = _cm_to_tnfpfn_tp(metrics.get("confusion_matrix"))

        row = {
            "report_path": str(rp),
            "dataset": dataset,
            "vtype": vtype,
            "classifier": cls,
            "run_id": run_id,
            "method": f"{vtype}+{cls}",

            "accuracy": metrics.get("accuracy"),
            "precision_macro": metrics.get("precision_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "f1_macro": metrics.get("f1_macro"),
            "roc_auc": metrics.get("roc_auc_macro") if "roc_auc_macro" in metrics else metrics.get("roc_auc"),

            "predicted_anomalies": metrics.get("predicted_anomalies"),
            "true_anomalies": metrics.get("true_anomalies"),

            "tn": tn, "fp": fp, "fn": fn, "tp": tp,

            # timing keys depend on your Timer; nbc_2.py stores these labels
            "time_total": times.get("Run time"),
            "time_vectorization": times.get("Vectorization"),
            "time_feature_extraction": times.get("Feature extraction"),
            "time_training": times.get("Training"),
            "time_inference": times.get("Inference"),

            "vec_params": json.dumps(vec_params, ensure_ascii=False),
            "cls_params": json.dumps(cls_params, ensure_ascii=False),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = [
        "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc",
        "predicted_anomalies", "true_anomalies",
        "tn", "fp", "fn", "tp",
        "time_total", "time_vectorization", "time_feature_extraction", "time_training", "time_inference",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ----------------------------- tables -----------------------------

def build_pivots(df: pd.DataFrame, outdir: Path) -> None:
    pivot_acc = pd.pivot_table(df, values="accuracy", index="dataset", columns="method", aggfunc="mean")
    pivot_f1 = pd.pivot_table(df, values="f1_macro", index="dataset", columns="method", aggfunc="mean")
    pivot_time = pd.pivot_table(df, values="time_total", index="dataset", columns="method", aggfunc="mean")

    pivot_acc.to_csv(outdir / "pivot_accuracy.csv", encoding="utf-8")
    pivot_f1.to_csv(outdir / "pivot_f1_macro.csv", encoding="utf-8")
    pivot_time.to_csv(outdir / "pivot_time_total.csv", encoding="utf-8")


def top_methods(df: pd.DataFrame, outdir: Path, top_n: int = 10) -> pd.DataFrame:
    agg = (
        df.groupby(["dataset", "method"], dropna=False)
          .agg(
              runs=("accuracy", "count"),
              f1_mean=("f1_macro", "mean"),
              acc_mean=("accuracy", "mean"),
              auc_mean=("roc_auc", "mean"),
              time_mean=("time_total", "mean"),
          )
          .reset_index()
          .sort_values(["dataset", "f1_mean", "acc_mean"], ascending=[True, False, False])
    )
    agg.to_csv(outdir / "method_ranking_by_dataset.csv", index=False, encoding="utf-8")

    global_rank = (
        df.groupby(["method"], dropna=False)
          .agg(
              runs=("accuracy", "count"),
              f1_mean=("f1_macro", "mean"),
              acc_mean=("accuracy", "mean"),
              auc_mean=("roc_auc", "mean"),
              time_mean=("time_total", "mean"),
          )
          .reset_index()
          .sort_values(["f1_mean", "acc_mean"], ascending=[False, False])
          .head(top_n)
    )
    global_rank.to_csv(outdir / "top_methods_global.csv", index=False, encoding="utf-8")
    return global_rank


# ----------------------------- plots -----------------------------

def plot_heatmap_accuracy(df: pd.DataFrame, outdir: Path) -> None:
    pivot = pd.pivot_table(df, values="accuracy", index="dataset", columns="method", aggfunc="mean")
    if pivot.empty:
        return

    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    mat = pivot.to_numpy()
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * pivot.shape[1]), max(4, 0.8 * pivot.shape[0])))
    im = ax.imshow(mat, aspect="auto")

    ax.set_title("Accuracy heatmap (mean) by dataset × method")
    ax.set_xlabel("Method (vtype+classifier)")
    ax.set_ylabel("Dataset")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=90)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")

    fig.tight_layout()
    fig.savefig(outdir / "heatmap_accuracy.png", dpi=200)
    plt.close(fig)


def plot_time_vs_accuracy(df: pd.DataFrame, outdir: Path) -> None:
    d = df.dropna(subset=["time_total", "accuracy"]).copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(d["time_total"].to_numpy(), d["accuracy"].to_numpy(), alpha=0.8)

    ax.set_title("Time vs Accuracy")
    ax.set_xlabel("Total time (s)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(outdir / "scatter_time_vs_accuracy.png", dpi=200)
    plt.close(fig)


def plot_f1_boxplot_by_vectorizer(df: pd.DataFrame, outdir: Path) -> None:
    d = df.dropna(subset=["f1_macro", "vtype"]).copy()
    if d.empty:
        return

    order = sorted(d["vtype"].unique().tolist())
    data = [d[d["vtype"] == vt]["f1_macro"].to_numpy() for vt in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=order, showmeans=True)
    ax.set_title("F1-score (macro) by vectorizer")
    ax.set_xlabel("Vectorizer (vtype)")
    ax.set_ylabel("F1 macro")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(outdir / "boxplot_f1_by_vectorizer.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrices(df: pd.DataFrame, outdir: Path) -> None:
    d = df.dropna(subset=["tn", "fp", "fn", "tp"]).copy()
    if d.empty:
        return

    d = d.sort_values(["dataset", "f1_macro", "accuracy"], ascending=[True, False, False])

    cm_dir = outdir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    for i, row in d.reset_index(drop=True).iterrows():
        cm = np.array(
            [[int(row["tn"]), int(row["fp"])],
             [int(row["fn"]), int(row["tp"])]],
            dtype=int
        )

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, aspect="equal")
        ax.set_title(f"{row['dataset']} / {row['method']}")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])

        for (r, c), val in np.ndenumerate(cm):
            ax.text(c, r, str(val), ha="center", va="center")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        fname = f"cm_{row['dataset']}_{row['vtype']}_{row['classifier']}.png"
        fig.savefig(cm_dir / fname, dpi=200)
        plt.close(fig)


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze experiment JSON reports and build tables/plots")
    ap.add_argument("--runs", default="runs", help="Root directory with runs/*/*/*/*/report-*.json")
    ap.add_argument("--outdir", default="analysis_out", help="Output directory for tables and plots")
    ap.add_argument("--top", type=int, default=10, help="Top-N methods/runs for outputs and confusion matrices")
    args = ap.parse_args()

    runs_root = Path(args.runs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_reports(runs_root)
    if df.empty:
        print(f"No reports found under: {runs_root.resolve()}")
        return

    df.to_csv(outdir / "all_results.csv", index=False, encoding="utf-8")

    build_pivots(df, outdir)

    top_global = top_methods(df, outdir, top_n=args.top)
    print("\nTop methods (global mean F1 then mean accuracy):")
    print(top_global[["method", "runs", "f1_mean", "acc_mean", "auc_mean", "time_mean"]].to_string(index=False))

    top_runs = df.sort_values(["f1_macro", "accuracy"], ascending=False).head(args.top)
    top_runs.to_csv(outdir / "top_runs.csv", index=False, encoding="utf-8")

    plot_heatmap_accuracy(df, outdir)
    plot_time_vs_accuracy(df, outdir)
    plot_f1_boxplot_by_vectorizer(df, outdir)
    plot_confusion_matrices(df, outdir)

    print(f"\nSaved outputs to: {outdir.resolve()}")
    print("Created: all_results.csv, pivot_*.csv, method_ranking_by_dataset.csv, top_methods_global.csv, top_runs.csv, *.png")


if __name__ == "__main__":
    main()
