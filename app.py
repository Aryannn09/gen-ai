"""Offline Streamlit dashboard for saved Forest Covertype results."""

import io
import json
import zipfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data import (
    ALL_FEATURES, CLASS_NAMES, CONTINUOUS_FEATURES, PROJECT_DIR,
    SOIL_FEATURES, WILDERNESS_FEATURES,
)

OUTPUTS = PROJECT_DIR / "outputs"
RESULTS_PATH = OUTPUTS / "all_results.joblib"
MODELS_PATH = OUTPUTS / "trained_models.joblib"
REGISTRY_PATH = OUTPUTS / "graph_registry.json"
METRICS_PATH = OUTPUTS / "metrics_comparison.csv"

st.set_page_config(page_title="Covertype Multiclass Classification", layout="wide")
st.title("Forest Covertype: Multiclass Classification Basics")
st.caption("Local 350,000-row stratified sample · 54 features · 7 classes · offline demo")


@st.cache_data
def load_results():
    return joblib.load(RESULTS_PATH)


@st.cache_resource
def load_models():
    return joblib.load(MODELS_PATH)


@st.cache_resource
def load_ovr_model():
    return joblib.load(OUTPUTS / "models" / "ovr_demo.joblib")


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_registry():
    with REGISTRY_PATH.open(encoding="utf-8") as file:
        return json.load(file)


if not RESULTS_PATH.is_file() or not MODELS_PATH.is_file():
    st.error("Saved results are missing. Run `python train.py` to train offline from data/covertype.csv.")
    if not (PROJECT_DIR / "data" / "covertype.csv").is_file():
        st.info(
            "The local dataset is missing. Prepare it once while online with:\n\n"
            "`from sklearn.datasets import fetch_covtype; "
            "fetch_covtype(as_frame=True).frame.to_csv('data/covertype.csv', index=False)`"
        )
    st.stop()

results = load_results()
models = load_models()
registry = load_registry()
graph_by_id = {graph["id"]: graph for graph in registry}

tabs = st.tabs([
    "Dataset Overview",
    "Graph Gallery",
    "EDA Graphs",
    "Train and Validate",
    "Metrics and Comparison",
    "Confusion Matrices",
    "Decision Boundaries",
    "Predict",
    "Theory",
])


def graph_path(graph):
    return PROJECT_DIR / graph["file_path"]


def show_registry_graph(graph, key_suffix=""):
    path = graph_path(graph)
    if path.is_file():
        st.image(str(path), width="stretch")
        st.caption(graph["one_line_explanation"])
        st.download_button(
            "Download graph",
            data=path.read_bytes(),
            file_name=path.name,
            mime="image/png",
            key=f"download_{key_suffix}_{graph['id']}",
        )
    else:
        st.warning(f"Registered graph file is missing: {graph['file_path']}")


with tabs[0]:
    st.header("Dataset Overview")
    overview = results["dataset_overview"]
    cols = st.columns(4)
    cols[0].metric("Sample rows", f"{overview['shape'][0]:,}")
    cols[1].metric("Source rows", f"{overview['source_rows']:,}")
    cols[2].metric("Features", len(ALL_FEATURES))
    cols[3].metric("Classes / missing", f"{len(CLASS_NAMES)} / {overview['missing_values']:,}")
    st.write("The sample is stratified from the local 581,012-row Forest Covertype CSV.")
    counts = overview["class_distribution"]["counts"]
    percentages = overview["class_distribution"]["percentages"]
    class_table = pd.DataFrame([
        {"Class": key, "Cover type": CLASS_NAMES[int(key)],
         "Rows": counts[key], "Percent": percentages[key]}
        for key in counts
    ])
    st.subheader("Class distribution (imbalanced)")
    st.dataframe(class_table, width="stretch", hide_index=True)
    types = overview["feature_types"]
    st.subheader("Feature types")
    st.write(f"{len(types['continuous'])} continuous features and {len(types['binary'])} binary one-hot features.")
    st.dataframe(pd.DataFrame({
        "Feature": ALL_FEATURES,
        "Type": ["Continuous" if name in CONTINUOUS_FEATURES else "Binary one-hot"
                 for name in ALL_FEATURES],
    }), width="stretch", hide_index=True)
    st.subheader("Summary statistics")
    st.dataframe(overview["summary_statistics"], width="stretch")
    st.subheader("Project objectives")
    st.markdown(
        "(a) Visualize feature relationships  ·  (b) Compare three required models on accuracy  ·  "
        "(c) Analyze errors with confusion matrices and per-class metrics  ·  "
        "(d) Validate generalization and overfitting  ·  (e) Visualize decision boundaries  ·  "
        "(f) Tune hyperparameters and plot search results  ·  (g) Recommend the best model."
    )

with tabs[1]:
    st.header("Graph Gallery")
    categories = [
        "EDA", "Validation", "Model Comparison", "Confusion Matrices",
        "ROC Curves", "Decision Boundaries", "Hyperparameter Tuning", "Class Imbalance",
    ]
    selected = st.selectbox("Filter graphs by category", ["Show all"] + categories)
    visible = registry if selected == "Show all" else [
        item for item in registry if item["category"] == selected
    ]
    st.write(f"{len(visible)} registered graph(s)")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for item in registry:
            path = graph_path(item)
            if path.is_file():
                zipped.write(path, arcname=path.name)
    st.download_button(
        "Download all graphs (ZIP)", archive.getvalue(),
        file_name="covertype_graphs.zip", mime="application/zip",
    )
    for category in categories:
        group = [item for item in visible if item["category"] == category]
        if group:
            st.subheader(category)
            for item in group:
                st.markdown(f"**{item['title']}**")
                show_registry_graph(item, "gallery")

with tabs[2]:
    st.header("EDA Graphs")
    for item in registry:
        if item["category"] in ("EDA", "Class Imbalance"):
            st.subheader(item["title"])
            show_registry_graph(item, "eda")

with tabs[3]:
    st.header("Train and Validate")
    metric_rows = []
    for name, row in results["metrics"].items():
        metric_rows.append({
            "Model": name,
            "Train accuracy": row["train_accuracy"],
            "Validation accuracy": row["val_accuracy"],
            "Train log loss": row["train_loss"],
            "Validation log loss": row["val_loss"],
            "Train-validation gap": row["train_validation_gap"],
            "Fit flag": row["overfitting_flag"],
            "5-fold CV": f"{results['cv_results'][name]['mean']:.4f} ± {results['cv_results'][name]['std']:.4f}",
        })
    st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)
    for item in registry:
        if item["category"] in ("Validation", "Hyperparameter Tuning"):
            st.subheader(item["title"])
            show_registry_graph(item, "validation")

with tabs[4]:
    st.header("Metrics and Comparison")
    metric_frame = load_metrics()
    st.dataframe(metric_frame, width="stretch", hide_index=True)
    st.download_button(
        "Download comparison CSV", metric_frame.to_csv(index=False).encode("utf-8"),
        file_name="metrics_comparison.csv", mime="text/csv",
    )
    for item in registry:
        if item["category"] in ("Model Comparison", "ROC Curves"):
            st.subheader(item["title"])
            show_registry_graph(item, "comparison")
    st.subheader("Per-class test metrics")
    model_choice = st.selectbox("Model for per-class metrics", list(models), key="per_class_model")
    class_metrics = results["per_class_metrics"][model_choice]
    st.dataframe(pd.DataFrame(class_metrics).T, width="stretch")
    st.subheader("Data-driven recommendation")
    st.success(results["recommendation"])

with tabs[5]:
    st.header("Confusion Matrices")
    cm_models = list(results["confusion_matrices"])
    model_choice = st.selectbox("Model", cm_models, key="cm_model")
    safe_name = model_choice.lower().replace(" ", "_")
    for kind in ("raw", "normalized"):
        graph_id = f"confusion_{safe_name}_{kind}"
        item = graph_by_id[graph_id]
        st.subheader(item["title"])
        show_registry_graph(item, "confusion")
    st.caption("Rows are true classes; columns are predicted classes; normalized values are row proportions.")

with tabs[6]:
    st.header("Decision Boundaries")
    item = graph_by_id["decision_boundaries"]
    st.image(str(graph_path(item)), width="stretch")
    st.caption(item["one_line_explanation"])

with tabs[7]:
    st.header("Predict a Cover Type")
    defaults = results["feature_defaults"]
    with st.form("prediction_form"):
        input_values = {}
        columns = st.columns(2)
        slider_specs = {
            "Elevation": (1800, 4000), "Aspect": (0, 360), "Slope": (0, 70),
            "Horizontal_Distance_To_Hydrology": (0, 1400),
            "Vertical_Distance_To_Hydrology": (-200, 700),
            "Horizontal_Distance_To_Roadways": (0, 7200),
            "Hillshade_9am": (0, 255), "Hillshade_Noon": (0, 255),
            "Hillshade_3pm": (0, 255),
            "Horizontal_Distance_To_Fire_Points": (0, 7200),
        }
        for i, (name, (low, high)) in enumerate(slider_specs.items()):
            value = int(np.clip(round(defaults[name]), low, high))
            input_values[name] = columns[i % 2].slider(
                name.replace("_", " "), low, high, value, key=f"input_{name}",
            )
        wilderness = st.selectbox(
            "Wilderness area", WILDERNESS_FEATURES,
            format_func=lambda feature: f"Wilderness area {WILDERNESS_FEATURES.index(feature) + 1}",
        )
        soil = st.selectbox(
            "Soil type", SOIL_FEATURES,
            format_func=lambda feature: f"Soil type {SOIL_FEATURES.index(feature) + 1}",
        )
        model_name = st.selectbox("Model", list(models))
        submitted = st.form_submit_button("Predict")
    if submitted:
        row = dict(defaults)
        row.update(input_values)
        for name in WILDERNESS_FEATURES + SOIL_FEATURES:
            row[name] = 0.0
        row[wilderness] = 1.0
        row[soil] = 1.0
        vector = np.asarray([[row[name] for name in results["feature_names"]]], dtype=np.float32)
        model = models[model_name]
        prediction = int(model.predict(vector)[0])
        st.success(f"Predicted class: {prediction} — {CLASS_NAMES[prediction]}")
        probabilities = model.predict_proba(vector)[0]
        st.dataframe(pd.DataFrame({
            "Class": [CLASS_NAMES[int(c)] for c in model.classes_],
            "Probability": probabilities,
        }).sort_values("Probability", ascending=False), width="stretch", hide_index=True)

with tabs[8]:
    st.header("Theory")
    from theory import get_all_theory_sections

    section_titles = {
        "Multiclass Introduction": "a) Multiclass vs Binary vs Multilabel",
        "OvR vs OvO": "b) One-vs-Rest and One-vs-One",
        "Softmax & Multinomial LR": "c) Softmax / Multinomial Logistic Regression",
        "Model Multiclass Handling": "d) Native Multiclass Model Strategies",
        "Model Intuitions": "e) Model Intuition, Hyperparameters, Strengths and Weaknesses",
        "Validation Theory": "f) Splits, Cross-Validation, Overfitting, Curves",
        "Evaluation Metrics": "g) Metric Definitions and Reading Results",
        "Decision Boundary Theory": "h) Decision Boundaries and 2D Limitations",
        "Dataset Notes": "i) Covertype Dataset Notes",
        "Jury Q&A": "k) Jury Q&A",
    }
    theory = get_all_theory_sections()
    for key, content in theory.items():
        title = section_titles.get(key, key)
        with st.expander(title, expanded=False):
            st.markdown(content)
    with st.expander("j) Final recommendation from actual test results", expanded=True):
        st.markdown(results["recommendation"])
    with st.expander("Live OvR Logistic Regression demo", expanded=False):
        st.write(
            "This fitted OneVsRestClassifier contains one binary Logistic Regression "
            "classifier per class. Choose a held-out example to inspect its class scores."
        )
        demo = results["ovr_demo"]
        row_index = st.slider("Held-out example", 0, len(demo["true_labels"]) - 1, 0)
        if st.button("Run OvR example"):
            ovr = load_ovr_model()
            probabilities = ovr.predict_proba(
                np.asarray([demo["example_features"][row_index]], dtype=np.float32)
            )[0]
            predicted = int(ovr.classes_[int(np.argmax(probabilities))])
            st.write(f"Predicted: **{CLASS_NAMES[predicted]}**; actual: **{CLASS_NAMES[int(demo['true_labels'][row_index])]}**")
            st.dataframe(pd.DataFrame({
                "Class": [CLASS_NAMES[int(c)] for c in ovr.classes_],
                "OvR score": probabilities,
            }).sort_values("OvR score", ascending=False), hide_index=True)
