from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from pptx import Presentation
from pptx.dml.color import RGBColor as PptRGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches as PptInches, Pt as PptPt


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
GRAPHS = OUTPUTS / "graphs"
SCREENSHOTS = SUBMISSION / "screenshots"
ASSETS = SCREENSHOTS / "step-evidence"
RESULTS = joblib.load(OUTPUTS / "all_results.joblib")
MODELS = joblib.load(OUTPUTS / "trained_models.joblib")
METRICS = RESULTS["metrics"]
CLASSES = RESULTS["class_names"]
NAVY = "17324D"
GREEN = "23836B"
TEAL = "3B8C92"
PALE = "EAF2F3"
INK = "243746"
MUTED = "5B6B75"


def graph(name: str) -> Path:
    return GRAPHS / name


def prepare_evidence() -> dict[str, Path]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    captures = SCREENSHOTS / "browser-captures"
    captures.mkdir(parents=True, exist_ok=True)
    for path in SCREENSHOTS.glob("0[1-9]-*.png"):
        destination = captures / path.name
        if path.exists() and not destination.exists():
            shutil.move(str(path), str(destination))

    evidence = {
        "dataset": graph("class_distribution.png"),
        "gallery": graph("pairplot_top_features.png"),
        "eda": graph("feature_boxplots.png"),
        "validation": graph("learning_curves.png"),
        "metrics": graph("accuracy_comparison.png"),
        "confusion": graph("confusion_random_forest_normalized.png"),
        "boundary": graph("decision_boundaries.png"),
    }

    defaults = RESULTS["feature_defaults"]
    row = pd.DataFrame([{name: defaults[name] for name in RESULTS["feature_names"]}])
    model = MODELS["Random Forest"]
    prediction_input = row.to_numpy()
    probabilities = model.predict_proba(prediction_input)[0]
    classes = model.classes_
    labels = [CLASSES[int(value)] for value in classes]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    order = np.argsort(probabilities)
    ax.barh(np.array(labels)[order], probabilities[order], color="#23836B")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted class probability")
    ax.set_title(f"Offline prediction example: {CLASSES[int(model.predict(prediction_input)[0])]}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    evidence["prediction"] = ASSETS / "prediction_example.png"
    fig.savefig(evidence["prediction"], dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.axis("off")
    labels = ["54 input\nfeatures", "7 class\nscores", "Softmax /\nOvR", "Predicted\ncover type"]
    xs = [0.12, 0.38, 0.64, 0.88]
    for index, (x, label) in enumerate(zip(xs, labels)):
        ax.text(
            x,
            0.54,
            label,
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            bbox={"boxstyle": "round,pad=0.8", "facecolor": "#17324D", "edgecolor": "#23836B"},
            transform=ax.transAxes,
        )
        if index < len(xs) - 1:
            ax.annotate(
                "",
                xy=(xs[index + 1] - 0.09, 0.54),
                xytext=(x + 0.09, 0.54),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "#23836B"},
            )
    ax.text(
        0.5,
        0.08,
        "OvR fits one binary classifier per class; softmax normalizes class scores to sum to 1.",
        ha="center",
        fontsize=10,
        color="#243746",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    evidence["theory"] = ASSETS / "multiclass_theory.png"
    fig.savefig(evidence["theory"], dpi=160, bbox_inches="tight")
    plt.close(fig)

    for name, path in evidence.items():
        if not path.is_file():
            raise FileNotFoundError(f"Evidence image is missing: {path}")
        if name not in {"prediction", "theory"}:
            shutil.copy2(path, ASSETS / f"{name}_{path.name}")
            evidence[name] = ASSETS / f"{name}_{path.name}"
    return evidence


def set_cell_shading(cell, fill: str) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shading)


def set_cell_text(cell, text: str, *, bold: bool = False, color: str = INK, size: float = 8.5) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(1)
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.name = "Aptos"
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    tr_pr.append(repeat)


def add_word_table(document, headers, rows, widths=None, font_size=8.5):
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    set_repeat_table_header(table.rows[0])
    for index, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[index], header, bold=True, color="FFFFFF", size=font_size)
        set_cell_shading(table.rows[0].cells[index], NAVY)
    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for index, value in enumerate(values):
            set_cell_text(cells[index], value, size=font_size)
            if row_index % 2:
                set_cell_shading(cells[index], "F2F6F7")
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    document.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_caption(document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(1)
    paragraph.paragraph_format.space_after = Pt(5)
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string(MUTED)


def add_bullet(document, text: str, level: int = 0) -> None:
    paragraph = document.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.left_indent = Inches(0.25 + 0.2 * level)
    run = paragraph.add_run(text)
    run.font.size = Pt(9.2)
    run.font.color.rgb = RGBColor.from_string(INK)


def add_heading(document, text: str, level: int = 1) -> None:
    paragraph = document.add_heading(text, level=level)
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(4)


def set_doc_defaults(document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    normal = document.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(9.2)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_after = Pt(4)
    for style_name, size, color in [("Heading 1", 16, NAVY), ("Heading 2", 12, GREEN), ("Heading 3", 10, NAVY)]:
        style = document.styles[style_name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header.add_run("FOREST COVERTYPE  |  FINAL JURY PROJECT")
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string(MUTED)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Editable project report  •  Page ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)


def add_page_title(document, title: str, subtitle: str | None = None) -> None:
    add_heading(document, title, 1)
    if subtitle:
        paragraph = document.add_paragraph(subtitle)
        paragraph.paragraph_format.space_after = Pt(6)
        for run in paragraph.runs:
            run.italic = True
            run.font.color.rgb = RGBColor.from_string(MUTED)


def figure_in_cell(cell, image: Path, width: float, caption: str):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(image), width=Inches(width))
    caption_p = cell.add_paragraph()
    caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_p.paragraph_format.space_after = Pt(1)
    run = caption_p.add_run(caption)
    run.italic = True
    run.font.size = Pt(7.5)


def build_report(evidence: dict[str, Path]) -> Path:
    doc = Document()
    set_doc_defaults(doc)

    # Page 1 — cover
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(85)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("MULTICLASS CLASSIFICATION BASICS")
    run.bold = True
    run.font.size = Pt(25)
    run.font.color.rgb = RGBColor.from_string(NAVY)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Model Making on the Forest Covertype Dataset")
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor.from_string(GREEN)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("FINAL JURY PROJECT REPORT")
    run.bold = True
    run.font.size = Pt(14)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(36)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Prepared by\nAsta  •  Aryan  •  Dhiraj  •  Ridhima").font.size = Pt(13)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(30)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Python  |  scikit-learn  |  pandas  |  Streamlit\nOffline training and inference  •  350,000-row stratified sample").font.size = Pt(11)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(75)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Editable report — update institution, course and team details before submission.").italic = True
    doc.add_page_break()

    # Page 2 — abstract and objectives
    add_page_title(doc, "1. Project Summary and Objectives")
    doc.add_paragraph(
        "This project builds an offline multiclass classification workflow for predicting one of seven forest cover types from cartographic and environmental measurements. "
        "A local Covertype CSV is validated, a reproducible stratified sample of 350,000 rows is selected, and four supervised models are trained and compared. "
        "The Streamlit application presents data exploration, validation, evaluation, two-feature decision boundaries, class probabilities and theory."
    )
    add_heading(doc, "Seven project objectives", 2)
    objectives = [
        "Visualise feature relationships with pair plots, correlation heatmaps, class counts and box plots.",
        "Compare Logistic Regression, kNN and Decision Tree accuracy; include Random Forest as an additional ensemble baseline.",
        "Analyse errors using raw and row-normalised confusion matrices and per-class precision, recall and F1.",
        "Assess generalisation with a stratified split, five-fold CV, learning/validation curves and train-validation gaps.",
        "Visualise model boundaries using Elevation and Horizontal_Distance_To_Roadways.",
        "Tune kNN and Decision Tree hyperparameters with bounded GridSearchCV experiments and plot the results.",
        "Recommend a model using measured test metrics, class imbalance and inference trade-offs.",
    ]
    for index, objective in enumerate(objectives, 1):
        add_bullet(doc, f"({chr(96 + index)}) {objective}")
    add_heading(doc, "Scope and constraints", 2)
    add_bullet(doc, "Runtime uses only local files and saved artifacts; there are no external API calls or runtime downloads.")
    add_bullet(doc, "Random state is fixed at 42; the source CSV is expected at data\\covertype.csv and is intentionally not bundled.")
    add_bullet(doc, "Deliverables include an interactive Streamlit app, notebook, model artifacts, CSV metrics, graph registry and this report/deck.")
    doc.add_page_break()

    # Page 3 — dataset
    add_page_title(doc, "2. Dataset, Sampling and Exploratory Analysis")
    doc.add_paragraph(
        "Forest Covertype contains 581,012 observations and 54 predictors describing wilderness, soil and terrain. "
        "The target has seven cover types. The experiment uses exactly 350,000 stratified rows so class proportions are retained while repeated analysis remains manageable."
    )
    add_word_table(
        doc,
        ["Profile item", "Observed value"],
        [
            ["Source / experiment rows", "581,012 / 350,000 (stratified)"],
            ["Predictors / target classes", "54 features / 7 cover types"],
            ["Feature types", "10 continuous; 44 binary one-hot indicators"],
            ["Missing values", str(RESULTS["dataset_overview"]["missing_values"])],
            ["Sampling and reproducibility", "Stratified sampling; random_state=42"],
        ],
        widths=[2.5, 4.0],
        font_size=8.5,
    )
    add_heading(doc, "Class distribution in the experiment sample", 2)
    class_counts = RESULTS["dataset_overview"]["class_distribution"]["counts"]
    class_percentages = RESULTS["dataset_overview"]["class_distribution"]["percentages"]
    class_rows = [
        [f"{key} — {CLASSES[key]}", f"{class_counts[key]:,}", f"{class_percentages[key]:.3f}%"]
        for key in CLASSES
    ]
    add_word_table(doc, ["Class", "Rows", "Sample share"], class_rows, widths=[3.5, 1.4, 1.4], font_size=8)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(evidence["dataset"]), width=Inches(4.7))
    add_caption(doc, "Class counts are imbalanced: class 2 is largest (170,660) and class 4 is smallest (1,655).")
    doc.add_page_break()

    # Page 4 — principles and architecture
    add_page_title(doc, "3. Design Principles and System Architecture")
    add_heading(doc, "Design principles", 2)
    for item in [
        "Offline-first: local CSV, local joblib artifacts and no network-dependent inference.",
        "Reproducible: fixed seeds, stratified sampling/splits and recorded evaluation results.",
        "Leakage-aware: fit preprocessing inside scikit-learn Pipelines using training folds only.",
        "Class-aware: report macro as well as weighted metrics and inspect per-class recall.",
        "Usable: nine ordered Streamlit tabs, cached data/resources and one-click graph downloads.",
        "Traceable: every generated plot is saved and indexed in one GRAPH_REGISTRY with a caption.",
    ]:
        add_bullet(doc, item)
    add_heading(doc, "Block diagram", 2)
    architecture_rows = [
        ["1. Local input", "data\\covertype.csv → validate required columns and target"],
        ["2. Sample and inspect", "350,000-row stratified sample → schema, missing values, summaries, EDA"],
        ["3. Split and model", "70/15/15 stratified train/validation/test → pipelines and bounded tuning"],
        ["4. Validate and compare", "5-fold CV, curves, metrics, confusion matrices, OvR ROC-AUC"],
        ["5. Save and present", "joblib models/results + CSV + registered graphs → offline Streamlit UI"],
    ]
    add_word_table(doc, ["Block", "Main responsibility / data flow"], architecture_rows, widths=[1.6, 4.9], font_size=8.7)
    add_heading(doc, "User-facing application", 2)
    doc.add_paragraph(
        "The GUI has exactly nine tabs in this order: Dataset Overview, Graph Gallery, EDA Graphs, Train and Validate, "
        "Metrics and Comparison, Confusion Matrices, Decision Boundaries, Predict, and Theory. "
        "The Gallery loads graphs exclusively from the registry and supports category filtering and downloads."
    )
    doc.add_page_break()

    # Page 5 — model details and theory
    add_page_title(doc, "4. Technical Model Details")
    add_word_table(
        doc,
        ["Model", "Multiclass handling", "Preprocessing / key controls"],
        [
            ["Logistic Regression", "Multinomial class probabilities (softmax)", "StandardScaler; regularization C; max_iter"],
            ["k-Nearest Neighbors", "Votes over the k nearest training examples", "StandardScaler; n_neighbors; distance metric"],
            ["Decision Tree", "One tree partitions feature space into class leaves", "No scaling required; max_depth; min_samples_leaf"],
            ["Random Forest", "Votes/probabilities aggregated over many trees", "No scaling required; tree count; depth/features per split"],
        ],
        widths=[1.35, 2.55, 2.6],
        font_size=8,
    )
    add_heading(doc, "Multiclass strategies", 2)
    doc.add_paragraph(
        "A multiclass task assigns exactly one label among K classes; multilabel prediction may assign several labels at once. "
        "One-vs-Rest (OvR) trains K binary classifiers, each separating one class from all others, then selects the strongest score. "
        "One-vs-One trains K(K−1)/2 pairwise classifiers and combines their votes. Logistic Regression can use a multinomial softmax objective; "
        "kNN, Decision Trees and Random Forests support multiple classes directly."
    )
    add_heading(doc, "Softmax and probability", 2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    formula = p.add_run("P(y = k | x) = exp(zₖ) / Σⱼ exp(zⱼ)     and     predicted class = argmaxₖ P(y = k | x)")
    formula.bold = True
    formula.font.size = Pt(11)
    doc.add_paragraph(
        "Here zₖ is the model score for class k. Softmax produces non-negative probabilities that sum to one. "
        "Scaling is important for distance-based kNN and gradient-based Logistic Regression, but tree splits are not scale-sensitive."
    )
    add_heading(doc, "Why Covertype is not Iris", 2)
    doc.add_paragraph(
        "Covertype has 54 mixed predictors, seven imbalanced classes and hundreds of thousands of examples. "
        "Cover types overlap in feature space and depend on nonlinear interactions; a small linear model that is adequate for a compact teaching dataset may underfit this setting."
    )
    doc.add_page_break()

    # Page 6 — validation
    add_page_title(doc, "5. Validation, Generalisation and Tuning")
    add_word_table(
        doc,
        ["Subset", "Rows", "Use"],
        [
            ["Train", "245,000 (70%)", "Fit pipeline/model parameters"],
            ["Validation", "52,500 (15%)", "Compare training configurations and inspect generalisation"],
            ["Test", "52,500 (15%)", "Final held-out performance only"],
        ],
        widths=[1.4, 1.6, 3.9],
        font_size=8.4,
    )
    cv_rows = []
    for name, result in RESULTS["cv_results"].items():
        cv_rows.append([name, f"{result['mean']:.4f}", f"{result['std']:.4f}"])
    add_word_table(doc, ["Model", "5-fold mean accuracy", "Std. dev."], cv_rows, widths=[2.8, 1.8, 1.5], font_size=8.2)
    gap_rows = [
        [name, f"{metrics['train_accuracy']:.3f}", f"{metrics['val_accuracy']:.3f}", f"{metrics['train_validation_gap']:+.3f}", metrics["overfitting_flag"]]
        for name, metrics in METRICS.items()
    ]
    add_word_table(doc, ["Model", "Train acc.", "Val. acc.", "Gap", "Flag"], gap_rows, widths=[2.0, 1.0, 1.0, 0.9, 1.3], font_size=7.8)
    image_table = doc.add_table(rows=1, cols=2)
    image_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    figure_in_cell(image_table.cell(0, 0), graph("train_validation_accuracy.png"), 3.05, "Training vs validation accuracy")
    figure_in_cell(image_table.cell(0, 1), graph("validation_curves.png"), 3.05, "Validation curves for kNN and Decision Tree")
    doc.add_paragraph(
        "Learning curves and five-fold stratified CV are also saved in outputs/graphs. The grid-search subsamples selected kNN n_neighbors=5 "
        "(CV accuracy 0.735) and Decision Tree max_depth=8 (CV accuracy 0.720); the validation-curve peaks are a separate curve experiment."
    )
    doc.add_page_break()

    # Page 7 — results
    add_page_title(doc, "6. Test Results and Error Analysis")
    comparison_rows = [
        [
            name,
            f"{m['test_accuracy']:.3f}",
            f"{m['precision_macro']:.3f}",
            f"{m['recall_macro']:.3f}",
            f"{m['f1_macro']:.3f}",
            f"{m['f1_weighted']:.3f}",
            f"{m['roc_auc_ovr']:.3f}",
            f"{m['test_loss']:.3f}",
        ]
        for name, m in METRICS.items()
    ]
    add_word_table(
        doc,
        ["Model", "Acc.", "Macro P", "Macro R", "Macro F1", "Weighted F1", "OvR AUC", "Log loss"],
        comparison_rows,
        widths=[1.1, 0.65, 0.75, 0.75, 0.75, 0.9, 0.7, 0.7],
        font_size=7.1,
    )
    doc.add_paragraph(
        "Training / test-prediction time (s): Logistic Regression 11.63 / 0.013; kNN 0.19 / 8.47; "
        "Decision Tree 3.81 / 0.012; Random Forest 5.62 / 0.178. Timings are specific to this run and machine."
    )
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(graph("precision_recall_f1.png")), width=Inches(5.3))
    add_caption(doc, "Random Forest has the highest test macro-F1 (0.854); kNN and Random Forest tie at 0.907 accuracy.")
    add_heading(doc, "Recommendation", 2)
    doc.add_paragraph(RESULTS["recommendation"])
    add_heading(doc, "Important caveat from class-wise error analysis", 2)
    rf_rows = []
    rf_report = RESULTS["per_class_metrics"]["Random Forest"]
    for name in CLASSES.values():
        row = rf_report[name]
        rf_rows.append([name, f"{row['precision']:.3f}", f"{row['recall']:.3f}", f"{row['f1-score']:.3f}", f"{int(row['support']):,}"])
    add_word_table(doc, ["Cover type", "Precision", "Recall", "F1", "Test support"], rf_rows, widths=[2.4, 1.0, 0.9, 0.8, 1.3], font_size=7.4)
    doc.add_paragraph(
        "Random Forest is flagged as overfit by the configured train-validation-gap rule (0.050) and has lower recall for rare Aspen (0.488) "
        "than for the dominant classes. The recommendation is a measured trade-off, not a claim that minority-class errors are solved."
    )
    doc.add_page_break()

    # Page 8 — evidence steps 1–3
    add_page_title(doc, "7. Working Application Evidence — Steps 1 to 3")
    doc.add_paragraph("The following figures correspond to the first three GUI steps and are generated from the saved offline experiment.")
    steps = [
        ("Step 1 — Dataset Overview", evidence["dataset"], "Class imbalance."),
        ("Step 2 — Graph Gallery", evidence["gallery"], "Pairwise class overlap."),
    ]
    image_table = doc.add_table(rows=1, cols=2)
    image_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, (title, image, caption) in enumerate(steps):
        figure_in_cell(image_table.cell(0, index), image, 3.0, f"{title}: {caption}")
    add_heading(doc, "Step 3 — EDA Graphs", 2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(evidence["eda"]), width=Inches(4.5))
    add_caption(doc, "Per-class distributions reveal feature spread and outliers.")
    doc.add_page_break()

    # Page 9 — evidence steps 4–6
    add_page_title(doc, "8. Working Application Evidence — Steps 4 to 6")
    steps = [
        ("Step 4 — Train and Validate", evidence["validation"], "Learning curves track data-size effects."),
        ("Step 5 — Metrics and Comparison", evidence["metrics"], "Accuracy must be read with macro metrics."),
    ]
    image_table = doc.add_table(rows=1, cols=2)
    image_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, (title, image, caption) in enumerate(steps):
        figure_in_cell(image_table.cell(0, index), image, 3.0, f"{title}: {caption}")
    add_heading(doc, "Step 6 — Confusion Matrices", 2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(evidence["confusion"]), width=Inches(3.6))
    add_caption(doc, "Row-normalised cells show per-class recall; Aspen is a remaining challenge.")
    doc.add_page_break()

    # Page 10 — evidence steps 7–9, contributions and run instructions
    add_page_title(doc, "9. Working Application Evidence — Steps 7 to 9")
    evidence_rows = [
        [evidence["boundary"], "Step 7 — Decision Boundaries\nTwo-dimensional view; other 52 predictors are not represented."],
        [evidence["prediction"], "Step 8 — Predict\nExample probabilities are computed by the saved Random Forest model."],
        [evidence["theory"], "Step 9 — Theory\nMulticlass scores are turned into a single predicted cover type."],
    ]
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, (image, caption) in enumerate(evidence_rows):
        cell = table.cell(0, index)
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(image), width=Inches(2.05))
        p = cell.add_paragraph(caption)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.size = Pt(7.1)
    add_heading(doc, "Proposed contribution allocation — verify before submission", 2)
    add_word_table(
        doc,
        ["Member", "Suggested contribution"],
        [
            ["Asta", "Dataset exploration, class imbalance analysis and EDA visualisations."],
            ["Aryan", "Data validation, preprocessing, model pipelines and training integration."],
            ["Dhiraj", "Cross-validation, hyperparameter search, metrics and error analysis."],
            ["Ridhima", "Streamlit interface, prediction/theory views, documentation and demo integration."],
        ],
        widths=[1.1, 5.4],
        font_size=7.7,
    )
    add_heading(doc, "Run steps and limitations", 2)
    doc.add_paragraph(
        "Install requirements; place the local CSV at data\\covertype.csv; run python train.py once; then run streamlit run app.py. "
        "For PowerShell, use python -m streamlit run app.py if the streamlit command is not on PATH. "
        "Limitations: minority classes remain difficult, kNN prediction is slower, two-feature boundaries are illustrative, "
        "and the local dataset CSV is required but not included in this folder."
    )
    output = SUBMISSION / "Forest_Covertype_Project_Report.docx"
    doc.save(output)
    return output


def ppt_text(slide, x, y, w, h, text, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(PptInches(x), PptInches(y), PptInches(w), PptInches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = PptInches(0.05)
    frame.margin_right = PptInches(0.05)
    frame.margin_top = PptInches(0.02)
    p = frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = "Aptos"
    run.font.size = PptPt(size)
    run.font.bold = bold
    run.font.color.rgb = PptRGBColor.from_string(color)
    return box


def add_slide_base(prs, title: str, page: int):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background.fill
    background.solid()
    background.fore_color.rgb = PptRGBColor(248, 250, 251)
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, PptInches(0.76))
    band.fill.solid()
    band.fill.fore_color.rgb = PptRGBColor.from_string(NAVY)
    band.line.fill.background()
    ppt_text(slide, 0.55, 0.12, 11.8, 0.48, title, size=24, color="FFFFFF", bold=True)
    ppt_text(slide, 0.55, 7.16, 11.8, 0.2, f"Forest Covertype • Final Jury • {page}", size=9, color=MUTED)
    return slide


def add_bullets(slide, x, y, w, h, items, size=17):
    box = ppt_text(slide, x, y, w, h, "", size=size)
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    for index, text in enumerate(items):
        p = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        p.text = f"• {text}"
        p.level = 0
        p.space_after = PptPt(10)
        p.font.name = "Aptos"
        p.font.size = PptPt(size)
        p.font.color.rgb = PptRGBColor.from_string(INK)
    return box


def add_ppt_table(slide, x, y, w, h, headers, rows, font_size=13):
    table_shape = slide.shapes.add_table(len(rows) + 1, len(headers), PptInches(x), PptInches(y), PptInches(w), PptInches(h))
    table = table_shape.table
    for col in table.columns:
        col.width = int(PptInches(w) / len(headers))
    for r_index, row in enumerate([headers] + rows):
        for c_index, value in enumerate(row):
            cell = table.cell(r_index, c_index)
            cell.text = str(value)
            cell.margin_left = PptInches(0.06)
            cell.margin_right = PptInches(0.05)
            cell.margin_top = PptInches(0.03)
            cell.margin_bottom = PptInches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = PptRGBColor.from_string(NAVY if r_index == 0 else ("FFFFFF" if r_index % 2 else PALE))
            for p in cell.text_frame.paragraphs:
                p.font.name = "Aptos"
                p.font.size = PptPt(font_size)
                p.font.bold = r_index == 0
                p.font.color.rgb = PptRGBColor.from_string("FFFFFF" if r_index == 0 else INK)
    return table_shape


def add_native_block(slide, x, y, w, h, text, fill=PALE, text_color=NAVY):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, PptInches(x), PptInches(y), PptInches(w), PptInches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = PptRGBColor.from_string(fill)
    shape.line.color.rgb = PptRGBColor.from_string(GREEN)
    shape.text = text
    shape.text_frame.word_wrap = True
    shape.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    for p in shape.text_frame.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        p.font.name = "Aptos"
        p.font.size = PptPt(15)
        p.font.bold = True
        p.font.color.rgb = PptRGBColor.from_string(text_color)
    return shape


def add_ppt_image(slide, path: Path, x: float, y: float, w: float, h: float):
    slide.shapes.add_picture(str(path), PptInches(x), PptInches(y), width=PptInches(w), height=PptInches(h))


def build_presentation(evidence: dict[str, Path]) -> Path:
    prs = Presentation()
    prs.slide_width = PptInches(13.333)
    prs.slide_height = PptInches(7.5)

    # Slide 1
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PptRGBColor.from_string(NAVY)
    ppt_text(slide, 0.9, 1.35, 11.6, 1.05, "MULTICLASS CLASSIFICATION BASICS", size=31, color="FFFFFF", bold=True, align=PP_ALIGN.CENTER)
    ppt_text(slide, 1.1, 2.6, 11.2, 0.8, "Model Making on the Forest Covertype Dataset", size=24, color="BFE4D7", align=PP_ALIGN.CENTER)
    ppt_text(slide, 2.0, 4.2, 9.3, 0.75, "Final Jury Examination • Offline ML Application", size=19, color="FFFFFF", align=PP_ALIGN.CENTER)
    ppt_text(slide, 2.0, 5.15, 9.3, 0.65, "Asta  |  Aryan  |  Dhiraj  |  Ridhima", size=18, color="FFFFFF", align=PP_ALIGN.CENTER)
    ppt_text(slide, 2.0, 6.45, 9.3, 0.3, "Editable PowerPoint — update institutional details before presenting", size=10, color="DCE6EA", align=PP_ALIGN.CENTER)

    # Slide 2
    slide = add_slide_base(prs, "Problem, scope and seven objectives", 2)
    add_bullets(slide, 0.7, 1.15, 6.1, 5.6, [
        "Predict one of seven forest cover types from 54 cartographic predictors.",
        "Use a reproducible, stratified 350,000-row sample from the local 581,012-row CSV.",
        "Explore relationships, imbalance and feature distributions.",
        "Compare Logistic Regression, kNN, Decision Tree and Random Forest.",
        "Validate with held-out data, five-fold CV, curves and targeted search.",
        "Analyse errors, visualise two-feature boundaries and recommend using actual test results.",
        "Deliver a fully offline Streamlit demo with saved models and results.",
    ], size=16)
    add_native_block(slide, 7.45, 1.55, 4.9, 1.05, "350,000 stratified rows")
    add_native_block(slide, 7.45, 3.0, 4.9, 1.05, "54 predictors • 7 cover types", fill="E7F1EC")
    add_native_block(slide, 7.45, 4.45, 4.9, 1.05, "Local data • no runtime APIs", fill="E8F0F6")

    # Slide 3
    slide = add_slide_base(prs, "Dataset profile and class imbalance", 3)
    add_ppt_image(slide, evidence["dataset"], 0.5, 1.0, 7.4, 5.7)
    add_bullets(slide, 8.1, 1.25, 4.55, 5.0, [
        "Source: Forest Covertype; 581,012 rows.",
        "Experiment: exactly 350,000 stratified rows; 10 continuous and 44 binary one-hot features.",
        "No missing values in the loaded dataset.",
        "Class 2: 170,660 rows (48.76%).",
        "Class 4: 1,655 rows (0.47%).",
        "Macro-F1 is important because accuracy can hide rare-class failures.",
    ], size=15)

    # Slide 4
    slide = add_slide_base(prs, "System architecture (editable block diagram)", 4)
    blocks = [
        ("Local CSV", "validate schema"),
        ("Stratified sample", "EDA + summary"),
        ("Split & pipelines", "train / tune"),
        ("Evaluate", "metrics + plots"),
        ("Offline app", "predict + explain"),
    ]
    xs = [0.45, 3.0, 5.55, 8.1, 10.65]
    for i, (title, subtitle) in enumerate(blocks):
        add_native_block(slide, xs[i], 2.25, 2.15, 1.65, f"{title}\n{subtitle}")
        if i < len(blocks) - 1:
            connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, PptInches(xs[i] + 2.18), PptInches(3.08), PptInches(xs[i + 1] - 0.08), PptInches(3.08))
            connector.line.color.rgb = PptRGBColor.from_string(GREEN)
            connector.line.width = PptPt(2)
            connector.line.end_arrowhead = True
    ppt_text(slide, 0.8, 4.7, 11.8, 1.0, "Artifacts: joblib models/results • comparison CSV • 24 registered graphs • notebook • Streamlit app", size=18, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    ppt_text(slide, 0.8, 5.75, 11.8, 0.7, "No dataset fetch, API call or network dependency is used at runtime.", size=16, color=GREEN, align=PP_ALIGN.CENTER)

    # Slide 5
    slide = add_slide_base(prs, "Models and multiclass strategies", 5)
    add_ppt_table(slide, 0.55, 1.05, 7.65, 4.85, ["Model", "How it handles 7 classes", "Scaling"], [
        ["Logistic Regression", "Multinomial softmax probabilities", "Yes"],
        ["kNN", "Neighbor vote / class probability", "Yes"],
        ["Decision Tree", "Multi-class leaves in one tree", "No"],
        ["Random Forest", "Aggregate probabilities over trees", "No"],
    ], font_size=13)
    ppt_text(slide, 8.55, 1.25, 4.0, 1.25, "OvR: K binary classifiers; each class vs all others.", size=17, color=INK, bold=True)
    ppt_text(slide, 8.55, 2.75, 4.0, 1.25, "OvO: K(K−1)/2 pairwise classifiers combined by voting.", size=17, color=INK, bold=True)
    ppt_text(slide, 8.55, 4.35, 4.0, 1.7, "Softmax:  P(y=k|x) = exp(zₖ) / Σⱼ exp(zⱼ)\nThe probabilities sum to 1; choose the largest.", size=15, color=GREEN, bold=True)
    ppt_text(slide, 0.7, 6.3, 11.8, 0.5, "Pipelines keep scaling inside training folds, preventing preprocessing leakage.", size=15, color=MUTED)

    # Slide 6
    slide = add_slide_base(prs, "Train, validate and tune", 6)
    add_ppt_image(slide, graph("learning_curves.png"), 0.45, 1.1, 6.15, 3.6)
    add_ppt_image(slide, graph("validation_curves.png"), 6.85, 1.1, 6.0, 3.6)
    add_bullets(slide, 0.7, 5.05, 12.0, 1.65, [
        "Stratified split: 70% train (245,000), 15% validation (52,500), 15% test (52,500).",
        "5-fold stratified CV: Random Forest 0.8158 ± 0.0032 on the configured CV subset.",
        "Grid search: kNN k=5 (0.735 CV accuracy); Decision Tree depth=8 (0.720).",
    ], size=14)

    # Slide 7
    slide = add_slide_base(prs, "Final test comparison and recommendation", 7)
    results_rows = [
        [
            name,
            f"{m['test_accuracy']:.3f}",
            f"{m['f1_macro']:.3f}",
            f"{m['f1_weighted']:.3f}",
            f"{m['roc_auc_ovr']:.3f}",
            f"{m['test_loss']:.3f}",
            f"{m['training_time_s']:.2f}",
            f"{m['prediction_time_s']:.2f}",
        ]
        for name, m in METRICS.items()
    ]
    add_ppt_table(slide, 0.55, 1.2, 8.5, 3.65, ["Model", "Acc.", "Macro F1", "Weighted F1", "OvR AUC", "Loss", "Fit s", "Pred s"], results_rows, font_size=11)
    add_native_block(slide, 9.45, 1.45, 3.3, 2.15, "Recommended:\nRandom Forest\nMacro-F1 = 0.854\nLog loss = 0.304", fill="E7F1EC")
    add_ppt_image(slide, graph("accuracy_comparison.png"), 0.65, 5.0, 6.0, 1.75)
    ppt_text(slide, 7.05, 5.0, 5.4, 1.4, "kNN and Random Forest tie on test accuracy (0.907). Random Forest wins on macro-F1 and log loss; its 0.050 train-validation gap triggers the project's overfit flag.", size=16, color=INK)

    # Slide 8
    slide = add_slide_base(prs, "Error analysis: confusion matrices and minority classes", 8)
    add_ppt_image(slide, graph("confusion_random_forest_normalized.png"), 0.45, 1.0, 6.0, 5.7)
    add_bullets(slide, 6.85, 1.35, 5.8, 4.8, [
        "Diagonal cells are correct predictions; off-diagonal cells show confusion.",
        "Row-normalised values make class recall comparable despite unequal supports.",
        "Random Forest Aspen recall is 0.488; rare classes remain challenging.",
        "Logistic Regression Aspen recall is only 0.008 in this test run.",
        "A recommendation must consider per-class risk, not accuracy alone.",
    ], size=16)

    # Slide 9
    slide = add_slide_base(prs, "Decision boundaries and feature importance", 9)
    add_ppt_image(slide, graph("decision_boundaries.png"), 0.35, 1.0, 8.05, 5.9)
    add_bullets(slide, 8.65, 1.45, 4.0, 4.7, [
        "Axes: Elevation and Horizontal_Distance_To_Roadways.",
        "Only two features are shown; the plot cannot encode the full 54-dimensional problem.",
        "Random Forest feature importance ranks Elevation highest in this run.",
        "Feature importance is association within this model, not causation.",
    ], size=15)

    # Slide 10
    slide = add_slide_base(prs, "Streamlit demo: end-to-end walkthrough", 10)
    steps = [
        ("1", "Dataset overview", "350k rows, schema, counts"),
        ("2", "Gallery / EDA", "registered charts and insights"),
        ("3", "Train and validate", "curves, CV and tuning"),
        ("4", "Compare / errors", "metrics and confusion matrices"),
        ("5", "Boundaries / predict", "2D plot and probabilities"),
        ("6", "Theory", "multiclass, OvR and jury Q&A"),
    ]
    for index, (number, title, detail) in enumerate(steps):
        col = index % 3
        row = index // 3
        add_native_block(slide, 0.7 + col * 4.2, 1.3 + row * 2.4, 3.65, 1.45, f"STEP {number}\n{title}\n{detail}", fill="EAF2F3")
    ppt_text(slide, 0.9, 6.35, 11.5, 0.35, "The app's nine tabs and all 24 output graphs are generated locally from saved results.", size=15, color=GREEN, align=PP_ALIGN.CENTER)

    # Slide 11
    slide = add_slide_base(prs, "Conclusion, limitations and jury takeaways", 11)
    add_bullets(slide, 0.75, 1.15, 11.8, 5.5, [
        "Best overall balance in this run: Random Forest, based on highest macro-F1 (0.854) and lowest test log loss (0.304).",
        "Accuracy is tied at 0.907 with kNN; kNN has slower full-test prediction time (8.47 s vs 0.18 s for Random Forest).",
        "Random Forest has a 0.050 train-validation gap and lower recall on rare Aspen; further calibration, class weighting or targeted data collection may help.",
        "Covertype is harder than Iris because of scale, 54 predictors, seven imbalanced classes and nonlinear overlap.",
        "The two-feature boundary is illustrative; final models use all 54 features.",
        "At runtime the project is offline, but users must supply the original CSV locally before the one-time training run.",
    ], size=17)

    # Slide 12
    slide = add_slide_base(prs, "Team contributions and run checklist", 12)
    add_ppt_table(slide, 0.55, 1.05, 6.15, 3.8, ["Member", "Proposed responsibility"], [
        ["Asta", "EDA, class distribution and visualization"],
        ["Aryan", "Data validation, pipelines and model training"],
        ["Dhiraj", "CV, tuning, metrics and error analysis"],
        ["Ridhima", "Streamlit UI, theory, docs and demo integration"],
    ], font_size=12)
    add_bullets(slide, 7.0, 1.15, 5.55, 4.6, [
        "1. Install: pip install -r requirements.txt",
        "2. Add data\\covertype.csv (local file; not bundled).",
        "3. Train once: python train.py",
        "4. Launch: python -m streamlit run app.py",
        "5. Verify the nine tabs, prediction, graph downloads and saved artifacts.",
        "Edit the proposed member allocation so it truthfully reflects each student's work.",
    ], size=15)
    ppt_text(slide, 0.7, 5.65, 12.0, 0.65, "Jury note: explain the split, macro-F1, minority-class errors, the overfit flag and why the model was recommended.", size=15, color=GREEN, bold=True, align=PP_ALIGN.CENTER)

    output = SUBMISSION / "Forest_Covertype_Final_Jury_Presentation.pptx"
    prs.save(output)
    return output


def main():
    evidence = prepare_evidence()
    report = build_report(evidence)
    presentation = build_presentation(evidence)
    readme = ROOT / "README.md"
    if readme.exists():
        shutil.copy2(readme, SUBMISSION / "README_Project_Run_Steps.md")
    metrics_csv = OUTPUTS / "metrics_comparison.csv"
    if metrics_csv.exists():
        shutil.copy2(metrics_csv, SUBMISSION / "Actual_Model_Comparison.csv")
    print(f"Report: {report}")
    print(f"Presentation: {presentation}")
    print(f"Evidence images: {len(list(ASSETS.glob('*.png')))}")


if __name__ == "__main__":
    main()
