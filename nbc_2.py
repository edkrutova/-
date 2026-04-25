#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Text classifier with configurable vectorizers + multiple classifiers.

Vectorizers:
  v1: Keras TextVectorization (count-like)
  v2: sklearn TF-IDF
  v3: Word2Vec embeddings
  v4: BERT embeddings
  v5: FastText embeddings

Classifiers:
  m  : MultinomialNB
  c  : ComplementNB
  rf : RandomForest
  svm: LinearSVC
  lr : Logistic Regression

Important:
  - MultinomialNB/ComplementNB require X >= 0. For embedding vectorizers (v3/v4/v5),
    we apply MinMaxScaler fitted ONLY on train to make features non-negative.
"""

import datetime
import json
import os
import sys
import numpy as np
import configparser
from sklearn.preprocessing import MinMaxScaler

from metrics_utils import MetricsCollector, Timer
from csvutils import load_data_and_labels_from_csv
from classifiers_2 import get_classifier, requires_nonnegative

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def save_results(filepath, metrics, timer, vec_params, cls_params):
    times = {name: timer.times[name]['duration'] for name in timer.times}
    with open(filepath, 'w') as f:
        report = {'metrics': metrics, 'times': times, 'vec_params': vec_params, 'cls_params': cls_params}
        json.dump(report, f, indent=4)
        f.write('\n')


def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')


def _cast_scalar(val: str):
    if not isinstance(val, str):
        return val
    s = val.strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        if s and s.lstrip("+-").isdigit():
            return int(s)
    except Exception:
        pass
    try:
        if any(ch in s for ch in (".", "e", "E")):
            return float(s)
    except Exception:
        pass
    return s


def cast_params(d):
    return {k: _cast_scalar(v) for k, v in (d or {}).items()}


def run(args):
    config = configparser.ConfigParser()
    config.read(args.config, encoding="utf-8")

    if "vectorizer" not in config or "classifier" not in config:
        print(f"ERROR: config {args.config} must contain [vectorizer] and [classifier] sections")
        sys.exit(1)

    vec_params = cast_params(dict(config["vectorizer"]))
    cls_params = cast_params(dict(config["classifier"]))

    cls_type = str(cls_params.get("type", "m")).lower()
    cls_params_no_type = {k: v for k, v in cls_params.items() if k != "type"}

    vtype = args.vtype.lower() if args.vtype else str(vec_params.pop("type", "v2")).lower()

    if vtype == "v1":
        from v1_1 import Vectorizer_keras as Vectorizer
        allowed_keys = {"vocab_size", "ngrams", "output_mode", "output_sequence_length"}
    elif vtype == "v2":
        from v2_1 import Vectorizer_sklearn_Tfidf as Vectorizer
        allowed_keys = {"max_features", "ngram_range"}
    elif vtype == "v3":
        from v3_1 import Vectorizer_Word2vec as Vectorizer
        allowed_keys = {"vector_size", "window", "min_count", "sg"}
    elif vtype == "v4":
        from v4_1 import Vectorizer_BERT as Vectorizer
        allowed_keys = {"model_name", "max_length"}
    elif vtype == "v5":
        from v5_1 import Vectorizer_FastText as Vectorizer
        allowed_keys = {"vector_size", "window", "min_count", "sg"}
    else:
        print(f"ERROR: Unknown vectorizer type {vtype}")
        sys.exit(1)

    vec_params = {k: v for k, v in vec_params.items() if k in allowed_keys}

    text_vectorizer = Vectorizer(**vec_params)
    print(f"Process dataset {args.dataset_name} with classifier {cls_type} > {text_vectorizer.info()}")

    metrics = MetricsCollector()
    timer = Timer()
    timer.start('Run time')

    dataset_dir = "/data/bel/RUDN/data"

    x_train, y_train = load_data_and_labels_from_csv(
        f"{dataset_dir}/{args.dataset_name}/train.csv",
        clean_mode=args.mode,
        header=[]
    )
    x_test, y_test = load_data_and_labels_from_csv(
        f"{dataset_dir}/{args.dataset_name}/test.csv",
        clean_mode=args.mode,
        header=[]
    )
    x_valid, y_valid = load_data_and_labels_from_csv(
        f"{dataset_dir}/{args.dataset_name}/validation.csv",
        clean_mode=args.mode,
        header=[]
    )

    timer.start('Vectorization')
    text_vectorizer.adapt(x_train)
    if args.verbose >= 2 and hasattr(text_vectorizer, "get_vocabulary"):
        try:
            print("Vocabulary sample:", text_vectorizer.get_vocabulary()[:50])
        except Exception:
            pass
    timer.stop('Vectorization')

    timer.start('Feature extraction')
    X_train = text_vectorizer.train_vector()
    X_test = text_vectorizer.transform(x_test)
    X_valid = text_vectorizer.transform(x_valid)
    timer.stop('Feature extraction')

    # NB requires non-negative features; embeddings (v3/v4/v5) can be negative.
    if requires_nonnegative(cls_type) and vtype in {"v3", "v4", "v5"}:
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_valid = scaler.transform(X_valid)

        X_train = np.clip(X_train, 0.0, None)
        X_test = np.clip(X_test, 0.0, None)
        X_valid = np.clip(X_valid, 0.0, None)

    classifier = get_classifier(cls_type, **cls_params_no_type)

    timer.start('Training')
    classifier.fit(X_train, y_train)
    timer.stop('Training')

    timer.start('Inference')
    y_pred = classifier.predict(X_test)
    timer.stop('Inference')

    timer.stop('Run time')

    y_score = None
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(X_test)
        if proba is not None and proba.shape[1] >= 2:
            y_score = proba[:, 1]
    elif hasattr(classifier, "decision_function"):
        y_score = classifier.decision_function(X_test)

    collected_metrics = metrics.evaluate(y_test, y_pred, y_score=y_score)
    metrics.print_metrics(collected_metrics, 'Metrics')
    timer.print_times()

    collected_metrics['confusion_matrix'] = collected_metrics['confusion_matrix'].tolist()
    config_dir = os.path.dirname(os.path.abspath(args.config))
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    
    out_path = os.path.join(config_dir, f"report-{config_name}-{timestamp()}.json")
    
    save_results(out_path, collected_metrics, timer, vec_params, cls_params)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Text classifier with configurable vectorizers + multiple classifiers")
    parser.add_argument("-d", "--dataset-name", default="100K", metavar="NAME", help="Dataset name")
    parser.add_argument("-m", "--mode", type=int, default=1, metavar="N", help="Clean input text mode N, default=1")
    parser.add_argument("-v", "--verbose", type=int, default=1, metavar="N", help="Verbose level")
    parser.add_argument("--config", required=True, help="Path to .cfg file with vectorizer and classifier params")
    parser.add_argument("--vtype", required=True, choices=["v1", "v2", "v3", "v4", "v5"], help="Vectorizer type")
    args = parser.parse_args()

    try:
        run(args)
    except KeyboardInterrupt:
        print("\n*** Interrupted by Ctrl-C ***")
