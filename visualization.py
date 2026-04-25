#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from csvutils import load_data_and_labels_from_csv
from classifiers_2 import get_classifier, requires_nonnegative


# ------------------------- utils -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sample_indices(n: int, k: int, seed: int) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False))


def load_vectorizer(vtype: str, vec_params: Dict[str, Any]):
    vtype = vtype.lower()
    if vtype == "v1":
        from v1_1 import Vectorizer_keras as Vectorizer
        allowed = {"vocab_size", "ngrams", "output_mode", "output_sequence_length"}
    elif vtype == "v2":
        from v2_1 import Vectorizer_sklearn_Tfidf as Vectorizer
        allowed = {"max_features", "ngram_range"}
    elif vtype == "v3":
        from v3_1 import Vectorizer_Word2vec as Vectorizer
        allowed = {"vector_size", "window", "min_count", "sg"}
    elif vtype == "v4":
        from v4_1 import Vectorizer_BERT as Vectorizer
        allowed = {"model_name", "max_length"}
    elif vtype == "v5":
        from v5_1 import Vectorizer_FastText as Vectorizer
        allowed = {"vector_size", "window", "min_count", "sg"}
    else:
        raise ValueError(f"Unknown vtype: {vtype}")

    filtered = {k: v for k, v in (vec_params or {}).items() if k in allowed}
    return Vectorizer(**filtered)


def best_effort_to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def parse_cfg(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    if "vectorizer" not in cfg:
        cfg["vectorizer"] = {}
    if "classifier" not in cfg:
        cfg["classifier"] = {}

    def cast(v: str):
        if not isinstance(v, str):
            return v
        s = v.strip()
        if s.lower() in {"true", "false"}:
            return s.lower() == "true"
        # int
        try:
            if s and s.lstrip("+-").isdigit():
                return int(s)
        except Exception:
            pass
        # float
        try:
            if any(ch in s for ch in (".", "e", "E")):
                return float(s)
        except Exception:
            pass
        return s

    vec_params = {k: cast(v) for k, v in dict(cfg["vectorizer"]).items()}
    cls_params = {k: cast(v) for k, v in dict(cfg["classifier"]).items()}
    return vec_params, cls_params


# ------------------------- t-SNE -------------------------

def run_tsne_for_vtype(
    dataset: str,
    data_dir: Path,
    vtype: str,
    vec_params: Dict[str, Any],
    outdir: Path,
    n_points: int,
    perplexity: float,
    seed: int,
) -> Path:
    x_train, y_train = load_data_and_labels_from_csv(str(data_dir / dataset / "train.csv"), clean_mode=1, header=[])
    x_test, y_test = load_data_and_labels_from_csv(str(data_dir / dataset / "test.csv"), clean_mode=1, header=[])

    x = np.array(list(x_train) + list(x_test), dtype=object)
    y = np.array(list(y_train) + list(y_test), dtype=int)

    idx = sample_indices(len(x), n_points, seed)
    x_s = x[idx].tolist()
    y_s = y[idx]

    vec = load_vectorizer(vtype, vec_params)
    vec.adapt(x_train)  # fit on train only (no leakage)
    X_s = vec.transform(x_s)
    X_s = best_effort_to_dense(X_s)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    Z = tsne.fit_transform(X_s)

    ensure_dir(outdir / "tsne")
    fig, ax = plt.subplots(figsize=(8, 6))

    mask0 = (y_s == 0)
    mask1 = (y_s == 1)
    ax.scatter(Z[mask0, 0], Z[mask0, 1], alpha=0.6, label="normal (0)", s=12)
    ax.scatter(Z[mask1, 0], Z[mask1, 1], alpha=0.8, label="anomaly (1)", s=14)

    ax.set_title(f"t-SNE: {dataset} / {vtype}")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    out_path = outdir / "tsne" / f"tsne_{dataset}_{vtype}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_examples_csv(path: Path, rows: List[Tuple[int, int, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "label", "text"])
        for r in rows:
            w.writerow(r)


def run_error_analysis(
    dataset: str,
    data_dir: Path,
    config_path: Path,
    vtype: str,
    cls_type: str,
    outdir: Path,
    k: int,
    seed: int,
) -> Tuple[Path, Path]:
    vec_params, cls_params = parse_cfg(config_path)
    cls_params = dict(cls_params)
    cls_params["type"] = cls_type

    x_train, y_train = load_data_and_labels_from_csv(str(data_dir / dataset / "train.csv"), clean_mode=1, header=[])
    x_test, y_test = load_data_and_labels_from_csv(str(data_dir / dataset / "test.csv"), clean_mode=1, header=[])

    vec = load_vectorizer(vtype, vec_params)
    vec.adapt(x_train)

    X_train = vec.train_vector() if hasattr(vec, "train_vector") else vec.transform(x_train)
    X_test = vec.transform(x_test)

    if requires_nonnegative(cls_type) and vtype in {"v3", "v4", "v5"}:
        scaler = MinMaxScaler()
        X_train_d = best_effort_to_dense(X_train)
        X_test_d = best_effort_to_dense(X_test)
        scaler.fit(X_train_d)
        X_train = np.clip(scaler.transform(X_train_d), 0.0, None)
        X_test = np.clip(scaler.transform(X_test_d), 0.0, None)

    cls_params_no_type = {k: v for k, v in cls_params.items() if k != "type"}
    clf = get_classifier(cls_type, **cls_params_no_type)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_true = np.asarray(y_test, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    fp_all = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_all = np.where((y_true == 1) & (y_pred == 0))[0]

    rng = np.random.default_rng(seed)
    fp_idx = fp_all if len(fp_all) <= k else rng.choice(fp_all, size=k, replace=False)
    fn_idx = fn_all if len(fn_all) <= k else rng.choice(fn_all, size=k, replace=False)

    fp_rows = [(int(i), int(y_true[i]), str(x_test[i])) for i in sorted(fp_idx)]
    fn_rows = [(int(i), int(y_true[i]), str(x_test[i])) for i in sorted(fn_idx)]

    err_dir = outdir / "errors"
    fp_path = err_dir / f"fp_{dataset}_{vtype}_{cls_type}.csv"
    fn_path = err_dir / f"fn_{dataset}_{vtype}_{cls_type}.csv"
    save_examples_csv(fp_path, fp_rows)
    save_examples_csv(fn_path, fn_rows)

    print(f"Error analysis: dataset={dataset} vtype={vtype} cls={cls_type}")
    print(f"Test size: {len(y_true)} | FP: {len(fp_all)} | FN: {len(fn_all)}")
    print(f"Saved FP examples: {fp_path}")
    print(f"Saved FN examples: {fn_path}")

    return fp_path, fn_path


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="t-SNE plots + error analysis")
    ap.add_argument("--dataset", default="1K", help="Dataset name (folder under data-dir)")
    ap.add_argument("--data-dir", default="/data/bel/RUDN/data", help="Root directory containing datasets")
    ap.add_argument("--outdir", default="plots", help="Output directory (plots/tsne, plots/errors)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # t-SNE
    ap.add_argument("--tsne-all", action="store_true", help="Run t-SNE for all vectorizers v1..v5")
    ap.add_argument("--vtypes", nargs="*", default=["v1", "v2", "v3", "v4", "v5"], help="Vectorizers for t-SNE")
    ap.add_argument("--tsne-n", type=int, default=1500, help="Number of points for t-SNE (sampled from train+test)")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")

    # Error analysis
    ap.add_argument("--config", default="universal_1.cfg", help="Base config file")
    ap.add_argument("--vtype", default="v5", help="Vectorizer for error analysis (v1..v5)")
    ap.add_argument("--classifier", default="lr", help="Classifier type (m/c/rf/svm/lr) for error analysis")
    ap.add_argument("--error-k", type=int, default=30, help="Max FP/FN examples to save")

    args = ap.parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    vec_params, _ = parse_cfg(Path(args.config))

    if args.tsne_all:
        for vt in args.vtypes:
            try:
                p = run_tsne_for_vtype(
                    dataset=args.dataset,
                    data_dir=data_dir,
                    vtype=vt,
                    vec_params=vec_params,
                    outdir=outdir,
                    n_points=args.tsne_n,
                    perplexity=args.perplexity,
                    seed=args.seed,
                )
                print(f"Saved t-SNE plot: {p}")
            except Exception as e:
                print(f"[WARN] t-SNE failed for {vt}: {e}")

    try:
        run_error_analysis(
            dataset=args.dataset,
            data_dir=data_dir,
            config_path=Path(args.config),
            vtype=args.vtype,
            cls_type=args.classifier,
            outdir=outdir,
            k=args.error_k,
            seed=args.seed,
        )
    except Exception as e:
        print(f"[WARN] Error analysis failed: {e}")


if __name__ == "__main__":
    main()
