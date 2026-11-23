"""Metrics, cross-validation, learning/validation curves, and error analysis."""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold, learning_curve, validation_curve, cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from data import CLASS_NAMES, stratified_subsample


def compute_all_metrics(trained_models, train_times, X_train, y_train,
                       X_val, y_val, X_test, y_test):
    classes = np.unique(y_train)
    results, class_reports = {}, {}
    for name, model in trained_models.items():
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        start = time.perf_counter()
        pred_test = model.predict(X_test)
        prediction_time = time.perf_counter() - start
        prob_train = model.predict_proba(X_train)
        prob_val = model.predict_proba(X_val)
        prob_test = model.predict_proba(X_test)
        m = {
            "model_name": name,
            "train_accuracy": accuracy_score(y_train, pred_train),
            "val_accuracy": accuracy_score(y_val, pred_val),
            "test_accuracy": accuracy_score(y_test, pred_test),
            "train_loss": log_loss(y_train, prob_train, labels=classes),
            "val_loss": log_loss(y_val, prob_val, labels=classes),
            "test_loss": log_loss(y_test, prob_test, labels=classes),
            "precision_macro": precision_score(y_test, pred_test, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, pred_test, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, pred_test, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_test, pred_test, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, pred_test, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, pred_test, average="weighted", zero_division=0),
            "roc_auc_ovr": roc_auc_score(
                y_test, prob_test, labels=classes, multi_class="ovr", average="macro",
            ),
            "training_time_s": train_times[name],
            "prediction_time_s": prediction_time,
        }
        gap = m["train_accuracy"] - m["val_accuracy"]
        m["train_validation_gap"] = gap
        m["overfitting_flag"] = (
            "Overfit" if gap > 0.05 else
            "Underfit" if m["train_accuracy"] < 0.65 and m["val_accuracy"] < 0.65 else
            "Good fit"
        )
        results[name] = m
        class_reports[name] = classification_report(
            y_test, pred_test, labels=classes,
            target_names=[CLASS_NAMES.get(int(c), str(c)) for c in classes],
            output_dict=True, zero_division=0,
        )
    return results, class_reports


def compute_cross_validation(pipelines, X, y, cv=5):
    X_sub, y_sub = stratified_subsample(X, y, min(20_000, len(y)))
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    result = {}
    for name, pipeline in pipelines.items():
        scores = cross_val_score(
            clone(pipeline), X_sub, y_sub, cv=splitter, scoring="accuracy", n_jobs=1,
        )
        result[name] = {
            "mean": float(scores.mean()), "std": float(scores.std()),
            "scores": scores.tolist(),
        }
    return result


def compute_learning_curves(pipelines, X, y):
    X_sub, y_sub = stratified_subsample(X, y, min(8_000, len(y)))
    result = {}
    for name, pipeline in pipelines.items():
        sizes, train, valid = learning_curve(
            clone(pipeline), X_sub, y_sub, train_sizes=np.linspace(.2, 1, 5),
            cv=3, scoring="accuracy", n_jobs=1, shuffle=True, random_state=42,
        )
        result[name] = {
            "train_sizes": sizes.tolist(),
            "train_mean": train.mean(axis=1).tolist(),
            "train_std": train.std(axis=1).tolist(),
            "val_mean": valid.mean(axis=1).tolist(),
            "val_std": valid.std(axis=1).tolist(),
        }
    return result


def compute_validation_curves(X, y):
    X_sub, y_sub = stratified_subsample(X, y, min(10_000, len(y)))
    models = {
        "kNN": (
            Pipeline([("scaler", StandardScaler()),
                      ("classifier", KNeighborsClassifier(n_jobs=-1))]),
            "classifier__n_neighbors", [3, 5, 9, 15],
        ),
        "Decision Tree": (
            Pipeline([("classifier", DecisionTreeClassifier(random_state=42))]),
            "classifier__max_depth", [5, 10, 15, 22],
        ),
    }
    results = {}
    for name, (pipeline, parameter, values) in models.items():
        train, valid = validation_curve(
            pipeline, X_sub, y_sub, param_name=parameter, param_range=values,
            cv=3, scoring="accuracy", n_jobs=1,
        )
        results[name] = {
            "param_values": values,
            "train_mean": train.mean(axis=1).tolist(),
            "train_std": train.std(axis=1).tolist(),
            "val_mean": valid.mean(axis=1).tolist(),
            "val_std": valid.std(axis=1).tolist(),
            "param_name": parameter.split("__")[-1],
        }
    return results


def get_confusion_matrices(trained_models, X_test, y_test):
    labels = sorted(np.unique(y_test))
    output = {}
    for name, model in trained_models.items():
        raw = confusion_matrix(y_test, model.predict(X_test), labels=labels)
        denominator = raw.sum(axis=1, keepdims=True)
        output[name] = {
            "raw": raw,
            "normalized": np.divide(raw, denominator, where=denominator != 0,
                                    out=np.zeros_like(raw, dtype=float)),
        }
    return output


def build_comparison_df(metrics_dict):
    return pd.DataFrame(list(metrics_dict.values()))


def generate_recommendation(metrics_dict):
    """Recommend by macro-F1 first, with test accuracy and time as context."""
    winner = max(metrics_dict, key=lambda n: (
        metrics_dict[n]["f1_macro"], metrics_dict[n]["test_accuracy"],
    ))
    best = metrics_dict[winner]
    return (
        f"{winner} is the recommended model because it has the highest test macro-F1 "
        f"({best['f1_macro']:.3f}); its test accuracy is {best['test_accuracy']:.3f}, "
        f"test log loss is {best['test_loss']:.3f}, and train-validation accuracy gap "
        f"is {best['train_validation_gap']:.3f}. Macro-F1 is prioritized because the "
        "Covertype classes are imbalanced; use the confusion matrices and per-class "
        "recall to decide whether minority-class performance meets the application need. "
        "Unlike Iris, Covertype has seven imbalanced, overlapping classes, 54 mixed "
        "features, and hundreds of thousands of rows, so simple linear boundaries or "
        "small toy-dataset intuition may not capture its nonlinear feature interactions."
    )
