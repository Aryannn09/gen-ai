"""Markdown explanations shared by the Covertype app and notebook."""


def get_multiclass_intro():
    return r"""
## Binary, multiclass, and multilabel classification

- **Binary:** choose one of two mutually exclusive labels (for example, spam/not spam).
- **Multiclass:** choose exactly one label from three or more mutually exclusive classes; Covertype has seven.
- **Multilabel:** assign any number of labels to one example (for example, several movie genres).

The classifier learns a mapping from the input features to one cover type. More classes, overlapping predictors, high dimensionality, and imbalance make the task harder than small balanced teaching examples such as Iris.
"""


def get_ovr_ovo_explanation():
    return r"""
## One-vs-Rest (OvR) and One-vs-One (OvO)

**OvR** fits K binary classifiers: class k versus every other class. Predict the class with the strongest score. A 3-class example with scores A-vs-rest = 0.75, B-vs-rest = 0.40, C-vs-rest = 0.30 predicts **A**. OvR needs K models but each binary task can be imbalanced, and independent scores are not automatically calibrated together.

**OvO** fits K(K−1)/2 pairwise classifiers. For three classes A, B, C, suppose A beats B, A beats C, and C beats B: A receives two votes and is predicted. With seven classes, OvO fits 21 classifiers. Each model sees only two classes, but the number of models grows quadratically.

| Strategy | Classifiers for K classes | Decision |
|---|---:|---|
| OvR | K | Highest class score |
| OvO | K(K−1)/2 | Majority vote |

This project includes a fitted `OneVsRestClassifier(LogisticRegression)` demo. Its scores are shown by class for a held-out example.
"""


def get_softmax_explanation():
    return r"""
## Softmax (multinomial) Logistic Regression

For class k, the model calculates a linear score (logit) `z_k = w_kᵀx + b_k`, then converts all K scores to probabilities:

$$P(y=k\mid x)=\frac{\exp(z_k)}{\sum_{j=1}^{K}\exp(z_j)}$$

The probabilities sum to one; the largest is the predicted class. Training minimizes multiclass cross-entropy:

$$L=-\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik}\log(p_{ik})$$

where `y_ik` is one for the true class and zero otherwise. The project uses scikit-learn Logistic Regression's multinomial behavior for multiclass labels and scales features in its pipeline.
"""


def get_model_multiclass_handling():
    return r"""
## How the project models handle multiple classes

| Model | Multiclass handling | Native? |
|---|---|---|
| Logistic Regression | Joint multinomial/softmax class probabilities; a separate OvR version is demonstrated | Multinomial yes |
| k-Nearest Neighbors | Votes among neighbors; probabilities are the class fractions among neighbors | Yes |
| Decision Tree | Each leaf stores class counts and probabilities; splits optimize multiclass impurity | Yes |
| Random Forest | Combines class votes/probabilities from many trees | Yes |

The training pipelines are Logistic Regression, kNN, Decision Tree, and Random Forest. The first two use `StandardScaler`; tree thresholds do not depend on feature scales.
"""


def get_model_intuitions():
    return r"""
## Model intuition, hyperparameters, strengths, and weaknesses

### Logistic Regression
Learns linear class scores and (for multiclass) softmax probabilities. Important settings: `C` (inverse regularization), `solver`, and `max_iter`. It is fast, stable, and probabilistic, but linear boundaries can miss nonlinear interactions.

### k-Nearest Neighbors (kNN)
A lazy learner that predicts from the labels of nearby training rows. `n_neighbors`, distance metric, and distance weighting control locality. It is intuitive and models nonlinear regions, but is scale-sensitive, memory-heavy, and prediction cost rises with the reference set.

### Decision Tree
Learns threshold rules that partition feature space. `max_depth`, `min_samples_leaf`, and `criterion` control complexity. Trees are interpretable and need no scaling, but deep trees can memorize training data and have high variance.

### Random Forest
Averages randomized decision trees to reduce variance and capture nonlinear interactions. `n_estimators`, `max_depth`, and `min_samples_leaf` control cost and complexity. It is robust and provides impurity-based feature importance, but is less interpretable and consumes more memory than one tree.
"""


def get_validation_theory():
    return r"""
## Splits, cross-validation, overfitting, and curves

The stratified **70/15/15** split assigns data to train (fit parameters), validation (compare/tune), and test (final evaluation). Stratification preserves class proportions in each split. The test set is not used for hyperparameter selection.

**Five-fold stratified cross-validation** trains on four folds and evaluates on the fifth, rotating the held-out fold; report mean ± standard deviation. For runtime, this project computes CV on a stratified 20,000-row training subset. Learning curves use a stratified subset of up to 8,000 rows; validation curves and grid search use up to 10,000 rows.

- **Overfitting:** training score is much better than validation score (large train-validation accuracy gap).
- **Underfitting:** both training and validation scores are low.
- **Learning curve:** score versus training sample size; a persistent gap suggests high variance, while both scores low suggests bias.
- **Validation curve:** training and validation scores versus a hyperparameter; use validation performance to find a useful complexity range.

The automatic fit flag uses the train-validation gap for overfitting and low train/validation accuracy for underfitting. CV and curves add complementary evidence.
"""


def get_metrics_explanation():
    return r"""
## Metrics and how to read them

- **Accuracy:** correct predictions divided by all examples. Simple to interpret, but can hide poor minority-class results.
- **Precision:** `TP/(TP+FP)`, the fraction of predictions for a class that are correct.
- **Recall:** `TP/(TP+FN)`, the fraction of actual examples of a class that are found.
- **F1:** harmonic mean of precision and recall. **Macro** averages per-class scores equally; **weighted** averages by class support.
- **OvR ROC-AUC:** area under each class-versus-rest ROC curve, macro-averaged here; higher means better ranking of positive examples.
- **Log loss:** negative log probability assigned to the true class; lower is better and confident wrong probabilities are penalized strongly.
- **Confusion matrix:** rows are true classes, columns are predictions. The diagonal is correct; off-diagonal cells show which classes are confused. A row-normalized version gives the fraction of each true class assigned to each prediction.

For imbalanced Covertype, inspect macro-F1 and per-class recall in addition to accuracy and weighted-F1.
"""


def get_decision_boundary_theory():
    return r"""
## Decision boundaries

A decision boundary separates regions assigned different predicted classes. Logistic Regression gives linear separators, kNN can form irregular local regions, and trees make axis-aligned regions.

The display uses **Elevation** and **Horizontal_Distance_To_Roadways** and fits the displayed models on a stratified 10,000-row sample. It overlays sample points. Two of 54 features cannot encode all soil, wilderness, and terrain interactions, so the 2-D view cannot fully separate all seven classes or represent the full-feature models' decision surface.
"""


def get_dataset_notes():
    return r"""
## Forest Covertype dataset notes

The dataset describes 30×30 m land patches in Roosevelt National Forest, Colorado, and predicts the dominant forest cover type.

- **Source CSV:** 581,012 rows, 54 predictors, and seven target classes (`Cover_Type`).
- **Features:** 10 continuous terrain/cartographic measures, 4 wilderness one-hot features, and 40 soil one-hot features. In the sklearn-exported CSV, their column suffixes are zero-based (`Wilderness_Area_0`…`_3`, `Soil_Type_0`…`_39`).
- **Classes:** 1 Spruce/Fir; 2 Lodgepole Pine; 3 Ponderosa Pine; 4 Cottonwood/Willow; 5 Aspen; 6 Douglas-fir; 7 Krummholz.
- **Imbalance:** classes 1 and 2 together account for about 85% of source rows; class 4 accounts for under 1%.

The project takes an **exactly 350,000-row stratified sample** so class proportions remain represented while multiple models and validation experiments remain practical. It is not assumed to produce identical results to training on all 581,012 rows.
"""


def get_jury_qa():
    return r"""
## Jury Q&A

### 1. How is multiclass handled by these models?
Logistic Regression uses joint softmax probabilities; kNN, Decision Tree, and Random Forest handle all classes natively. OvR is separately demonstrated for Logistic Regression.

### 2. Why can simple models suffice on Iris, and how does Covertype differ?
Iris is small, balanced, and relatively separable with four features. Covertype has 581,012 source rows, seven imbalanced classes, 54 mixed features, and more overlap and nonlinear interactions. Results must be measured, not inferred from Iris.

### 3. Why use OvR?
It extends a binary model with K classifiers and is easy to understand; subproblems may be imbalanced and independent scores may not be jointly calibrated.

### 4. Why scale features for kNN?
Distance calculations are dominated by features with larger numeric ranges if scaling is omitted. Scaling also improves the conditioning of Logistic Regression optimization.

### 5. Why is a Decision Tree prone to overfitting?
Deep splits can isolate and memorize training examples. Limit depth or increase leaf-size constraints; validate the result.

### 6. Why use log loss?
It evaluates predicted probabilities and penalizes confident wrong predictions, which hard-label accuracy cannot distinguish.

### 7. Why macro-F1 for imbalanced data?
It weights every class equally, so poor minority-class F1 is not hidden by high support classes. Weighted-F1 complements it by reflecting the observed frequency mix.

### 8. How do you know the model is not overfitting?
Compare training and validation accuracy/loss, inspect learning curves and five-fold CV variability, and report the held-out test result. No single score proves generalization.

### 9. Why sample 350,000 rows?
It preserves the class mix while limiting the cost of fitting and repeated validation, especially for kNN. Full-data performance may differ.

### 10. What are project limitations?
The experiment uses one fixed split and bounded search grids; the imbalance is not corrected; the baselines are limited; and 2-D boundaries omit 52 features. Results may not transfer to other forests or regions.
"""


def get_all_theory_sections():
    return {
        "Multiclass Introduction": get_multiclass_intro(),
        "OvR vs OvO": get_ovr_ovo_explanation(),
        "Softmax & Multinomial LR": get_softmax_explanation(),
        "Model Multiclass Handling": get_model_multiclass_handling(),
        "Model Intuitions": get_model_intuitions(),
        "Validation Theory": get_validation_theory(),
        "Evaluation Metrics": get_metrics_explanation(),
        "Decision Boundary Theory": get_decision_boundary_theory(),
        "Dataset Notes": get_dataset_notes(),
        "Jury Q&A": get_jury_qa(),
    }
