"""Scikit-learn model pipelines and bounded hyperparameter search."""

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_pipelines(random_state=42):
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                solver="lbfgs", max_iter=300, random_state=random_state,
            )),
        ]),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ]),
        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(max_depth=15, random_state=random_state)),
        ]),
        "Random Forest": Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=100, max_depth=24, min_samples_leaf=2,
                n_jobs=-1, random_state=random_state,
            )),
        ]),
    }


def build_ovr_pipeline(random_state=42):
    from sklearn.multiclass import OneVsRestClassifier

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", OneVsRestClassifier(
            LogisticRegression(solver="lbfgs", max_iter=200,
                               random_state=random_state),
            n_jobs=-1,
        )),
    ])


def train_models(pipelines, X_train, y_train):
    fitted, times = {}, {}
    for name, pipeline in pipelines.items():
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        times[name] = time.perf_counter() - start
        fitted[name] = pipeline
        print(f"[train] {name}: {times[name]:.1f}s", flush=True)
    return fitted, times


def hyperparameter_search(X_train, y_train, random_state=42):
    """Small 3-fold grid searches on a stratified 10k-row subset."""
    from data import stratified_subsample

    X_sub, y_sub = stratified_subsample(X_train, y_train, min(10_000, len(y_train)),
                                        random_state)
    cv = 3
    searches = {
        "kNN": (
            Pipeline([("scaler", StandardScaler()),
                      ("classifier", KNeighborsClassifier(n_jobs=-1))]),
            {"classifier__n_neighbors": [5, 10, 15]},
        ),
        "Decision Tree": (
            Pipeline([("classifier", DecisionTreeClassifier(random_state=random_state))]),
            {"classifier__max_depth": [8, 15, 22]},
        ),
    }
    results = {}
    for name, (pipeline, parameters) in searches.items():
        search = GridSearchCV(
            pipeline, parameters, scoring="accuracy", cv=cv, n_jobs=-1,
            refit=True, return_train_score=False,
        )
        search.fit(X_sub, y_sub)
        parameter_name = next(iter(parameters))
        bare_name = parameter_name.split("__")[-1]
        results[name] = {
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "param_name": bare_name,
            "param_values": [int(v) for v in search.cv_results_["param_" + parameter_name].data],
            "mean_scores": search.cv_results_["mean_test_score"].tolist(),
            "std_scores": search.cv_results_["std_test_score"].tolist(),
        }
        print(f"[search] {name}: {search.best_params_}, CV={search.best_score_:.4f}")
    return results


def get_feature_importance(pipeline, feature_names):
    estimator = pipeline.steps[-1][1]
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("Feature importance requires a fitted tree-based estimator.")
    importances = np.asarray(estimator.feature_importances_)
    order = np.argsort(importances)[::-1]
    return [(feature_names[i], float(importances[i])) for i in order]
