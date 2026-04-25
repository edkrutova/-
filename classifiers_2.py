#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""classifiers.py

Supported types:
  - 'm'   : MultinomialNB
  - 'c'   : ComplementNB
  - 'rf'  : RandomForestClassifier
  - 'svm' : LinearSVC
  - 'lr'  : LogisticRegression

Notes:
  - MultinomialNB/ComplementNB require non-negative features (X >= 0).
  - Most params arrive as strings from configparser; pass already-cast values.
"""

from __future__ import annotations

from typing import Any

from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

NB_TYPES = {"m", "c"}

def requires_nonnegative(cls_type: str) -> bool:
    return (cls_type or "").lower() in NB_TYPES

def get_classifier(cls_type: str, **params: Any):
    t = (cls_type or "").lower().strip()

    if t == "m":
        allowed = {"alpha", "fit_prior", "class_prior"}
        p = {k: v for k, v in params.items() if k in allowed}
        return MultinomialNB(**p)

    if t == "c":
        allowed = {"alpha", "fit_prior", "class_prior", "norm"}
        p = {k: v for k, v in params.items() if k in allowed}
        return ComplementNB(**p)

    if t == "rf":
        allowed = {
            "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
            "max_features", "class_weight", "bootstrap", "random_state", "n_jobs",
        }
        p = {k: v for k, v in params.items() if k in allowed}
        p.setdefault("n_estimators", 200)
        p.setdefault("random_state", 42)
        p.setdefault("n_jobs", -1)
        return RandomForestClassifier(**p)

    if t == "svm":
        allowed = {"C", "class_weight", "max_iter", "random_state", "dual", "loss"}
        p = {k: v for k, v in params.items() if k in allowed}
        p.setdefault("random_state", 42)
        p.setdefault("max_iter", 5000)
        return LinearSVC(**p)

    if t == "lr":
        allowed = {"C", "penalty", "solver", "class_weight", "max_iter", "random_state", "n_jobs"}
        p = {k: v for k, v in params.items() if k in allowed}
        p.setdefault("random_state", 42)
        p.setdefault("max_iter", 2000)
        return LogisticRegression(**p)

    raise ValueError(f"Unknown classifier type: {cls_type!r}. Use one of: m, c, rf, svm, lr")
