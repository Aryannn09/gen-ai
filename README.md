# Multiclass Classification Basics: Forest Covertype

An offline, reproducible scikit-learn project for classifying the seven Forest
Covertype classes. Training reads only `data/covertype.csv`; **there are no
network requests or external APIs at runtime**.

## Requirements and run

Use Python 3.10 or later:

```powershell
pip install -r requirements.txt
python train.py
streamlit run app.py
```

The training run can take a while because it evaluates multiple models and
validation workflows on 350,000 rows. It creates cached models, metrics,
tables, and plots under `outputs/`. Later app launches use those saved artifacts.

### Prepare the dataset once

The CSV is intentionally not bundled and is never fetched by the project.
If `data/covertype.csv` is absent, obtain it once while online with:

```python
from sklearn.datasets import fetch_covtype
fetch_covtype(as_frame=True).frame.to_csv("data/covertype.csv", index=False)
```

After the CSV has been created, `python train.py` and `streamlit run app.py`
work offline. The project itself does not import or call `fetch_covtype`.

## What is included

- Exactly 350,000 stratified rows and a stratified 70/15/15 train/validation/test
  split; all sampled rows are drawn from the local source CSV.
- Logistic Regression, kNN, Decision Tree, and Random Forest pipelines;
  scaling is applied to Logistic Regression and kNN.
- Train/validation accuracy and log loss, five-fold stratified CV, learning and
  validation curves, test metrics and confusion matrices, and a fitted OvR demo.
- Model-comparison CSV, actual-result-based recommendation, and all graphs saved
  in `outputs/graphs/` and indexed in `outputs/graph_registry.json`.
- A Streamlit app with nine tabs: Dataset Overview, Graph Gallery, EDA Graphs,
  Train and Validate, Metrics and Comparison, Confusion Matrices, Decision
  Boundaries, Predict, and Theory.
- A Jupyter notebook with explanations and runnable access to the same pipeline.

## Project files

`data.py`, `models.py`, `evaluate.py`, and `plots.py` hold reusable pipeline
components. `train.py` generates outputs once; `app.py` displays saved results.
`theory.py` supplies the Theory tab content. The generated `outputs/` artifacts
are not required in source control and are recreated by `python train.py`.

The project reports accuracy alongside macro and weighted precision/recall/F1,
ROC-AUC (one-vs-rest), log loss, runtime, and per-class errors. Macro-F1 is used
to rank recommendations because the dataset is imbalanced; the final selection
and its explanation are computed from the actual test results, not assumed in
advance. The 2-feature boundary plot is a visualization only, not the 54-feature
model's full decision surface.
