#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
experiment_runner.py

Как выбирается:
- векторизатор: аргумент --vtype (v1..v5)
- классификатор: [classifier] type в cfg (m/c/rf/svm/lr)

Скрипт:
- читает base cfg
- для каждой комбинации создаёт отдельный cfg в папке эксперимента
- запускает nbc_2.py
- складывает stdout/stderr в log.txt
- сохраняет все report-*.json (которые создаёт nbc_2.py) в папку эксперимента
"""

from __future__ import annotations

import argparse
import configparser
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime


VECTORIZERS_DEFAULT = ["v1", "v2", "v3", "v4", "v5"]
CLASSIFIERS_DEFAULT = ["m", "c", "rf", "svm", "lr"]


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_cfg(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    with path.open("r", encoding="utf-8") as f:
        cfg.read_file(f)

    if "vectorizer" not in cfg:
        cfg["vectorizer"] = {}
    if "classifier" not in cfg:
        cfg["classifier"] = {}
    return cfg


def write_cfg(cfg: configparser.ConfigParser, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        cfg.write(f)


def apply_classifier_overrides(cfg: configparser.ConfigParser, cls_type: str, balanced: bool) -> None:
    """
    Записываем classifier type и (опционально) class_weight=balanced для lr/svm/rf.
    NB (m/c) class_weight не поддерживает — туда не пишем.
    """
    cfg["classifier"]["type"] = cls_type

    if balanced and cls_type in {"lr", "svm", "rf"}:
        cfg["classifier"]["class_weight"] = "balanced"
    else:
        if "class_weight" in cfg["classifier"]:
            del cfg["classifier"]["class_weight"]


def run_one(
    python_bin: str,
    script_path: Path,
    dataset: str,
    vtype: str,
    cls_type: str,
    base_cfg_path: Path,
    out_root: Path,
    balanced: bool,
    extra_args: list[str],
) -> tuple[int, Path]:

    tag = now_tag()
    run_dir = out_root / dataset / vtype / cls_type / tag
    safe_mkdir(run_dir)

    cfg = load_cfg(base_cfg_path)
    apply_classifier_overrides(cfg, cls_type, balanced=balanced)
    cfg_path = run_dir / "config.cfg"
    write_cfg(cfg, cfg_path)

    cmd = [
        python_bin,
        str(script_path),
        "-d", dataset,
        "--config", str(cfg_path),
        "--vtype", vtype,
        *extra_args,
    ]

    log_path = run_dir / "log.txt"
    env = os.environ.copy()

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write("CMD: " + " ".join(cmd) + "\n\n")
        logf.flush()
        p = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        rc = p.returncode

    cwd = Path.cwd()
    for rep in sorted(cwd.glob("report-*.json")):
        target = run_dir / rep.name
        if not target.exists():
            try:
                shutil.move(str(rep), str(target))
            except Exception:
                shutil.copy2(str(rep), str(target))

    return rc, run_dir


def main():
    ap = argparse.ArgumentParser(description="Run all dataset×vectorizer×classifier experiments")
    ap.add_argument("--script", default="./nbc_2.py", help="Path to main experiment script (nbc_2.py)")
    ap.add_argument("--base-config", default="universal_1.cfg", help="Base config file")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    ap.add_argument("--outdir", default="runs", help="Output root directory")
    ap.add_argument("--datasets", nargs="*", default=["1K"], help="Datasets list, e.g. 1K 10K 100K")
    ap.add_argument("--vectorizers", nargs="*", default=VECTORIZERS_DEFAULT, help="Vectorizers list (v1..v5)")
    ap.add_argument("--classifiers", nargs="*", default=CLASSIFIERS_DEFAULT, help="Classifiers list (m c rf svm lr)")
    ap.add_argument("--balanced", action="store_true", help="Add class_weight=balanced for lr/svm/rf")
    ap.add_argument("--extra", nargs="*", default=[], help="Extra args appended to nbc_2.py call")
    args = ap.parse_args()

    script_path = Path(args.script)
    base_cfg = Path(args.base_config)
    out_root = Path(args.outdir)

    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}")
        sys.exit(2)
    if not base_cfg.exists():
        print(f"ERROR: base config not found: {base_cfg}")
        sys.exit(2)

    total = len(args.datasets) * len(args.vectorizers) * len(args.classifiers)
    print(f"Planned experiments: {total} "
          f"({len(args.datasets)} datasets × {len(args.vectorizers)} vectorizers × {len(args.classifiers)} classifiers)")
    print(f"Output: {out_root.resolve()}")
    safe_mkdir(out_root)

    ok = 0
    fail = 0

    for ds in args.datasets:
        for vtype in args.vectorizers:
            for cls_type in args.classifiers:
                print(f"\n=== RUN ds={ds} vtype={vtype} cls={cls_type} ===")
                rc, run_dir = run_one(
                    python_bin=args.python,
                    script_path=script_path,
                    dataset=ds,
                    vtype=vtype,
                    cls_type=cls_type,
                    base_cfg_path=base_cfg,
                    out_root=out_root,
                    balanced=args.balanced,
                    extra_args=args.extra,
                )
                if rc == 0:
                    ok += 1
                    print(f"OK  -> {run_dir}")
                else:
                    fail += 1
                    print(f"FAIL(rc={rc}) -> {run_dir} (see log.txt)")

    print(f"\nDONE. ok={ok} fail={fail} total={total}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
