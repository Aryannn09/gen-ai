"""One-time, fully offline training and artifact-generation entry point."""

import time
from pathlib import Path

import joblib

from data import (
    CLASS_NAMES, PROJECT_DIR, get_dataset_overview,
    get_decision_boundary_data, get_plotting_subsample, load_and_sample,
    split_data, stratified_subsample,
)
from evaluate import (
    build_comparison_df, compute_all_metrics, compute_cross_validation,
    compute_learning_curves, compute_validation_curves, generate_recommendation,
    get_confusion_matrices,
)
from models import (
    build_ovr_pipeline, build_pipelines, get_feature_importance,
    hyperparameter_search, train_models,
)
from plots import (
    GRAPH_REGISTRY, plot_accuracy_comparison, plot_class_distribution,
    plot_confusion_matrices, plot_correlation_heatmap, plot_cross_validation,
    plot_decision_boundaries, plot_feature_boxplots, plot_feature_importance,
    plot_learning_curves, plot_pairplot, plot_precision_recall_f1,
    plot_roc_curves, plot_search_results, plot_time_comparison,
    plot_train_val_accuracy, plot_train_val_loss, plot_validation_curves,
    reset_registry, save_registry,
)

OUTPUTS = PROJECT_DIR / "outputs"
MODELS_DIR = OUTPUTS / "models"
RANDOM_STATE = 42


def main():
    started = time.perf_counter()
    OUTPUTS.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    (OUTPUTS / "graphs").mkdir(exist_ok=True)
    reset_registry()
    print("Loading local Forest Covertype CSV and sampling 350,000 rows...", flush=True)
    df = load_and_sample(random_state=RANDOM_STATE)
    overview = get_dataset_overview(df)
    print(f"Sample shape: {overview['shape']}; missing cells: {overview['missing_values']}")

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = split_data(df)
    print(f"70/15/15 split: {len(y_train):,}/{len(y_val):,}/{len(y_test):,}")

    pipelines = build_pipelines(random_state=RANDOM_STATE)
    trained_models, training_times = train_models(pipelines, X_train, y_train)
    feature_importances = get_feature_importance(
        trained_models["Random Forest"], feature_names,
    )
    top_features = [name for name, _ in feature_importances[:5]]

    metrics, per_class_metrics = compute_all_metrics(
        trained_models, training_times, X_train, y_train,
        X_val, y_val, X_test, y_test,
    )
    cv_results = compute_cross_validation(build_pipelines(), X_train, y_train, cv=5)
    learning_results = compute_learning_curves(build_pipelines(), X_train, y_train)
    validation_results = compute_validation_curves(X_train, y_train)
    search_results = hyperparameter_search(X_train, y_train, random_state=RANDOM_STATE)
    confusion_matrices = get_confusion_matrices(trained_models, X_test, y_test)
    recommendation = generate_recommendation(metrics)

    plot_class_distribution(df)
    plot_pairplot(get_plotting_subsample(df, 5_000), top_features)
    plot_correlation_heatmap(df)
    plot_feature_boxplots(df)
    plot_feature_importance(feature_importances)
    plot_train_val_accuracy(metrics)
    plot_train_val_loss(metrics)
    plot_learning_curves(learning_results)
    plot_validation_curves(validation_results)
    plot_cross_validation(cv_results)
    plot_accuracy_comparison(metrics)
    plot_precision_recall_f1(metrics)
    plot_time_comparison(metrics)
    plot_roc_curves(trained_models, X_test, y_test)
    plot_confusion_matrices(confusion_matrices)
    plot_search_results(search_results)

    X_boundary, y_boundary = get_decision_boundary_data(df, n=10_000)
    boundary_models, _ = train_models(build_pipelines(), X_boundary, y_boundary)
    plot_decision_boundaries(boundary_models, X_boundary, y_boundary)

    # A compact fitted OvR Logistic Regression model powers the GUI's live demo.
    X_ovr, y_ovr = stratified_subsample(X_train, y_train, min(7_000, len(y_train)))
    ovr_model = build_ovr_pipeline(random_state=RANDOM_STATE)
    ovr_model.fit(X_ovr, y_ovr)
    demo_X = X_test[:20]
    ovr_demo = {
        "predictions": ovr_model.predict(demo_X),
        "probabilities": ovr_model.predict_proba(demo_X),
        "true_labels": y_test[:20],
        "example_features": demo_X,
    }

    comparison = build_comparison_df(metrics)
    comparison.to_csv(OUTPUTS / "metrics_comparison.csv", index=False)
    save_registry()
    results = {
        "dataset_overview": overview,
        "class_names": CLASS_NAMES,
        "feature_names": feature_names,
        "feature_defaults": dict(zip(
            feature_names,
            [float(v) for v in df[feature_names].median().to_numpy()],
        )),
        "metrics": metrics,
        "per_class_metrics": per_class_metrics,
        "cv_results": cv_results,
        "learning_curves": learning_results,
        "validation_curves": validation_results,
        "search_results": search_results,
        "confusion_matrices": confusion_matrices,
        "recommendation": recommendation,
        "graph_registry": GRAPH_REGISTRY.copy(),
        "ovr_demo": ovr_demo,
    }
    joblib.dump(trained_models, MODELS_DIR / "trained_models.joblib", compress=3)
    joblib.dump(trained_models, OUTPUTS / "trained_models.joblib", compress=3)
    joblib.dump(boundary_models, MODELS_DIR / "boundary_models.joblib", compress=3)
    joblib.dump(ovr_model, MODELS_DIR / "ovr_demo.joblib", compress=3)
    joblib.dump(results, OUTPUTS / "all_results.joblib", compress=3)
    print("\nModel comparison:")
    print(comparison[[
        "model_name", "test_accuracy", "f1_macro", "f1_weighted", "test_loss",
        "roc_auc_ovr", "training_time_s", "prediction_time_s",
    ]].to_string(index=False))
    print("\nRecommendation:", recommendation)
    print(f"Registered {len(GRAPH_REGISTRY)} graphs.")
    print(f"Finished in {(time.perf_counter() - started) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
