"""Saved visualizations and the single registry used by the app and notebook."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

from data import CLASS_NAMES, CONTINUOUS_FEATURES, stratified_subsample

ROOT = Path(__file__).resolve().parent
GRAPHS_DIR = ROOT / "outputs" / "graphs"
GRAPH_REGISTRY = []
PALETTE = sns.color_palette("tab10", 7)


def reset_registry():
    GRAPH_REGISTRY.clear()


def _save(fig, graph_id, category, title, filename, explanation):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPHS_DIR / filename
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    GRAPH_REGISTRY.append({
        "id": graph_id,
        "category": category,
        "title": title,
        "file_path": path.relative_to(ROOT).as_posix(),
        "one_line_explanation": explanation,
    })
    return str(path)


def save_registry(path=None):
    path = Path(path) if path else ROOT / "outputs" / "graph_registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(GRAPH_REGISTRY, file, indent=2)
    return str(path)


def plot_class_distribution(df):
    counts = df["Cover_Type"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{i}: {CLASS_NAMES[int(i)]}" for i in counts.index]
    bars = ax.bar(labels, counts.values, color=PALETTE)
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values], padding=3)
    ax.set(title="Cover Type Class Distribution (Stratified 350k Sample)",
           ylabel="Rows", xlabel="Cover type")
    ax.tick_params(axis="x", rotation=24)
    return _save(fig, "class_distribution", "Class Imbalance",
                 "Class Distribution", "class_distribution.png",
                 f"Class counts are imbalanced: class 2 is largest ({counts.max():,}) and class 4 is smallest ({counts.min():,}).")


def plot_pairplot(df, top_features):
    names = list(top_features[:5])
    plot_df = df[names + ["Cover_Type"]].copy()
    plot_df["Cover type"] = plot_df["Cover_Type"].map(CLASS_NAMES)
    grid = sns.pairplot(
        plot_df.drop(columns="Cover_Type"), hue="Cover type",
        vars=names, diag_kind="hist", corner=True,
        plot_kws={"s": 8, "alpha": .35, "linewidth": 0},
        diag_kws={"element": "step", "stat": "density"},
        palette=dict(zip(CLASS_NAMES.values(), PALETTE)),
    )
    grid.fig.suptitle("Top Feature Pair Plots by Cover Type (5,000-row sample)", y=1.02)
    path = _save(grid.fig, "pairplot", "EDA", "Pair Plot of Important Features",
                 "pairplot_top_features.png",
                 f"Class-colored pairwise distributions show overlap among the top features ({', '.join(names[:3])}) and help assess cover-type separability.")
    return path


def plot_correlation_heatmap(df):
    columns = CONTINUOUS_FEATURES
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(df[columns].corr(), cmap="vlag", center=0, square=True, ax=ax)
    ax.set_title("Correlation Heatmap (Continuous Features)")
    return _save(fig, "correlation_heatmap", "EDA", "Correlation Heatmap",
                 "correlation_heatmap.png",
                 "Correlations highlight redundant continuous predictors and feature relationships.")


def plot_feature_boxplots(df):
    sampled = pd.concat([
        part.sample(min(len(part), 1200), random_state=42)
        for _, part in df.groupby("Cover_Type")
    ])
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    for ax, feature in zip(axes.flat, CONTINUOUS_FEATURES):
        sns.boxplot(data=sampled, x="Cover_Type", y=feature, ax=ax, color="#77aadd",
                    showfliers=False)
        ax.set_title(feature, fontsize=9)
        ax.set_xlabel("Cover type")
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Continuous Feature Distributions by Class (sampled)", y=1.01)
    return _save(fig, "feature_boxplots", "EDA", "Feature Box Plots",
                 "feature_boxplots.png",
                 "Per-class box plots compare continuous-feature ranges and reveal overlap and outliers.")


def plot_feature_importance(importances):
    items = importances[:20][::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([x[0] for x in items], [x[1] for x in items], color="#4c956c")
    ax.set(title="Random Forest Feature Importance (Top 20)", xlabel="Impurity-based importance")
    return _save(fig, "feature_importance", "EDA", "Feature Importance",
                 "feature_importance.png",
                 f"Random Forest ranks {importances[0][0]} as the most influential feature; importance is predictive association, not causation.")


def plot_train_val_accuracy(metrics):
    names = list(metrics)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - .18, [metrics[n]["train_accuracy"] for n in names], .36, label="Train")
    ax.bar(x + .18, [metrics[n]["val_accuracy"] for n in names], .36, label="Validation")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_title("Training vs Validation Accuracy")
    best = max(names, key=lambda n: metrics[n]["val_accuracy"])
    return _save(fig, "train_val_accuracy", "Validation", "Train vs Validation Accuracy",
                 "train_validation_accuracy.png",
                 f"{best} has the highest validation accuracy ({metrics[best]['val_accuracy']:.3f}); gaps indicate generalization risk.")


def plot_train_val_loss(metrics):
    names = list(metrics)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - .18, [metrics[n]["train_loss"] for n in names], .36, label="Train")
    ax.bar(x + .18, [metrics[n]["val_loss"] for n in names], .36, label="Validation")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("Log loss (lower is better)")
    ax.legend()
    ax.set_title("Training vs Validation Log Loss")
    return _save(fig, "train_val_loss", "Validation", "Train vs Validation Log Loss",
                 "train_validation_log_loss.png",
                 "Lower validation log loss indicates better probability quality; a train-validation gap can signal overfitting.")


def plot_learning_curves(results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (name, curve) in zip(axes.flat, results.items()):
        sizes = curve["train_sizes"]
        for key, label, color in [
            ("train", "Training", "#4c72b0"), ("val", "Cross-validation", "#dd8452"),
        ]:
            mean = np.asarray(curve[key + "_mean"])
            std = np.asarray(curve[key + "_std"])
            ax.plot(sizes, mean, marker="o", label=label, color=color)
            ax.fill_between(sizes, mean - std, mean + std, alpha=.15, color=color)
        ax.set(title=name, xlabel="Training rows", ylabel="Accuracy", ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.suptitle("Learning Curves (stratified subsample, 3-fold CV)")
    best_name = max(results, key=lambda n: results[n]["val_mean"][-1])
    return _save(fig, "learning_curves", "Validation", "Learning Curves",
                 "learning_curves.png",
                 f"Training and cross-validation trends show data-size effects; {best_name} has the highest final cross-validation score ({results[best_name]['val_mean'][-1]:.3f}).")


def plot_validation_curves(curves):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, curve) in zip(axes, curves.items()):
        x = curve["param_values"]
        ax.plot(x, curve["train_mean"], "o-", label="Training")
        ax.plot(x, curve["val_mean"], "s--", label="Cross-validation")
        ax.set(title=f"{name}: {curve['param_name']}", xlabel=curve["param_name"],
               ylabel="Accuracy", ylim=(0, 1))
        ax.legend()
    insights = []
    for name, curve in curves.items():
        best_index = int(np.argmax(curve["val_mean"]))
        insights.append(
            f"{name} validation peaks at {curve['param_name']}="
            f"{curve['param_values'][best_index]} ({curve['val_mean'][best_index]:.3f})"
        )
    return _save(fig, "validation_curves", "Hyperparameter Tuning",
                 "Validation Curves (kNN and Decision Tree)", "validation_curves.png",
                 "; ".join(insights) + ".")


def plot_cross_validation(cv):
    names = list(cv)
    means = [cv[n]["mean"] for n in names]
    stds = [cv[n]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, means, yerr=stds, capsize=5, color=PALETTE[:len(names)])
    ax.set(title="5-Fold Stratified Cross-Validation Accuracy", ylabel="Accuracy", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, "cross_validation_scores", "Validation", "5-Fold Cross-Validation",
                 "cross_validation_scores.png",
                 "Bars show mean accuracy and standard deviation across five stratified folds; lower variance indicates more stable scores.")


def plot_accuracy_comparison(metrics):
    names = list(metrics)
    scores = [metrics[n]["test_accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, scores, color=PALETTE[:len(names)])
    ax.bar_label(bars, fmt="%.3f")
    ax.set(title="Final Test Accuracy Comparison", ylabel="Accuracy", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=20)
    best = names[int(np.argmax(scores))]
    return _save(fig, "accuracy_comparison", "Model Comparison", "Test Accuracy Comparison",
                 "accuracy_comparison.png",
                 f"{best} has the highest test accuracy at {max(scores):.3f}; accuracy should be read alongside class-balanced metrics.")


def plot_precision_recall_f1(metrics):
    names = list(metrics)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = .18
    for i, (metric, label) in enumerate([
        ("precision_macro", "Macro precision"), ("recall_macro", "Macro recall"),
        ("f1_macro", "Macro F1"), ("f1_weighted", "Weighted F1"),
    ]):
        ax.bar(x + (i - 1.5) * width, [metrics[n][metric] for n in names],
               width, label=label)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Precision, Recall and F1 Comparison")
    ax.legend()
    best = max(names, key=lambda name: metrics[name]["f1_macro"])
    return _save(fig, "prf_comparison", "Model Comparison",
                 "Precision / Recall / F1 Comparison", "precision_recall_f1.png",
                 f"Macro scores weight every class equally; {best} has the highest test macro-F1 ({metrics[best]['f1_macro']:.3f}), while weighted F1 follows class frequencies.")


def plot_time_comparison(metrics):
    names = list(metrics)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - .18, [metrics[n]["training_time_s"] for n in names], .36, label="Training")
    ax.bar(x + .18, [metrics[n]["prediction_time_s"] for n in names], .36, label="Test prediction")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("Seconds")
    ax.set_title("Training and Prediction Time")
    ax.legend()
    fastest = min(names, key=lambda name: metrics[name]["training_time_s"])
    return _save(fig, "time_comparison", "Model Comparison", "Training and Prediction Time",
                 "training_prediction_time.png",
                 f"Bars compare training and full-test prediction time; {fastest} trained fastest ({metrics[fastest]['training_time_s']:.2f}s) on this run.")


def plot_roc_curves(models, X_test, y_test):
    X_sample, y_sample = stratified_subsample(X_test, y_test, min(8_000, len(y_test)))
    classes = np.unique(y_sample)
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, model in models.items():
        probabilities = model.predict_proba(X_sample)
        per_class = []
        for index, label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_sample == label, probabilities[:, index])
            per_class.append((fpr, tpr))
        grid = np.linspace(0, 1, 250)
        mean_tpr = np.mean([np.interp(grid, fpr, tpr) for fpr, tpr in per_class], axis=0)
        ax.plot(grid, mean_tpr, label=f"{name} (macro AUC={auc(grid, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Macro One-vs-Rest ROC Curves (stratified test subset)")
    ax.legend(fontsize=8)
    return _save(fig, "roc_curves", "ROC Curves", "Macro OvR ROC Curves",
                 "roc_curves.png",
                 "Macro one-vs-rest ROC curves compare class-ranking quality across models on a stratified test subset.")


def plot_confusion_matrices(matrices):
    for name, pair in matrices.items():
        normalized = pair["normalized"]
        worst_class_index = int(np.argmin(np.diag(normalized)))
        worst_class = sorted(CLASS_NAMES)[worst_class_index]
        insight = (
            f"Class {worst_class} ({CLASS_NAMES[worst_class]}) has the lowest "
            f"per-class recall for {name} ({normalized[worst_class_index, worst_class_index]:.3f}); "
            "diagonal cells are correct and off-diagonal cells show errors."
        )
        for kind in ("raw", "normalized"):
            matrix = pair[kind]
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(matrix, cmap="Blues", annot=True,
                        fmt="d" if kind == "raw" else ".2f",
                        xticklabels=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
                        yticklabels=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)], ax=ax)
            ax.set(xlabel="Predicted class", ylabel="True class",
                   title=f"{name}: {kind.title()} Confusion Matrix")
            safe = name.lower().replace(" ", "_")
            _save(fig, f"confusion_{safe}_{kind}", "Confusion Matrices",
                  f"{name} - {kind.title()} Confusion Matrix",
                  f"confusion_{safe}_{kind}.png",
                  f"{kind.title()} confusion matrix for {name}; {insight}")


def plot_decision_boundaries(models, X, y):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 110),
                         np.linspace(y_min, y_max, 110))
    draw_X, draw_y = X, y
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (name, model) in zip(axes.flat, models.items()):
        regions = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, regions, alpha=.25, levels=np.arange(0.5, 8.5, 1),
                    cmap="tab10")
        ax.scatter(draw_X[:, 0], draw_X[:, 1], c=draw_y, s=5, alpha=.45,
                   cmap="tab10", vmin=1, vmax=7)
        ax.set(title=name, xlabel="Elevation", ylabel="Horizontal Distance to Roadways")
    fig.suptitle("2-Feature Decision Boundaries (10,000-row stratified model sample)")
    return _save(fig, "decision_boundaries", "Decision Boundaries",
                 "Decision Boundaries - Elevation vs Roadways",
                 "decision_boundaries.png",
                 "These 2-D model boundaries and sample points use Elevation and distance to roadways; they cannot represent all 54-feature interactions.")


def plot_search_results(search_results):
    fig, axes = plt.subplots(1, len(search_results), figsize=(11, 4))
    if len(search_results) == 1:
        axes = [axes]
    for ax, (name, result) in zip(axes, search_results.items()):
        x = result["param_values"]
        ax.plot(x, result["mean_scores"], "o-")
        ax.set(title=name, xlabel=result["param_name"], ylabel="3-fold CV accuracy")
    fig.suptitle("Grid Search Results (10,000-row stratified subset)")
    insights = []
    for name, result in search_results.items():
        best_index = int(np.argmax(result["mean_scores"]))
        insights.append(
            f"{name} peaks at {result['param_name']}="
            f"{result['param_values'][best_index]} "
            f"(CV accuracy {result['mean_scores'][best_index]:.3f})"
        )
    return _save(fig, "search_results", "Hyperparameter Tuning",
                 "Grid Search Results", "hyperparameter_search.png",
                 "; ".join(insights) + ".")
