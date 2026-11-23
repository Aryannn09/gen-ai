"""Offline Covertype loading, stratified sampling, and dataset summaries."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET_COLUMN = "Cover_Type"
CONTINUOUS_FEATURES = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS_FEATURES = [f"Wilderness_Area_{i}" for i in range(4)]
SOIL_FEATURES = [f"Soil_Type_{i}" for i in range(40)]
BINARY_FEATURES = WILDERNESS_FEATURES + SOIL_FEATURES
ALL_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES
DECISION_BOUNDARY_FEATURES = ["Elevation", "Horizontal_Distance_To_Roadways"]
CLASS_NAMES = {
    1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine",
    4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas-fir", 7: "Krummholz",
}
PROJECT_DIR = Path(__file__).resolve().parent


def _resolve_path(filepath):
    path = Path(filepath)
    return path if path.is_absolute() else PROJECT_DIR / path


def load_and_sample(filepath="data/covertype.csv", n_samples=350_000,
                    random_state=RANDOM_STATE):
    """Load the local CSV and return exactly n_samples stratified rows."""
    path = _resolve_path(filepath)
    if not path.is_file():
        raise FileNotFoundError(
            f"Covertype CSV was not found: {path}\n"
            "Create data/covertype.csv once while online with:\n"
            "from sklearn.datasets import fetch_covtype\n"
            "fetch_covtype(as_frame=True).frame.to_csv('data/covertype.csv', index=False)\n"
            "After that, training and the app run fully offline."
        )
    df = pd.read_csv(path)
    missing_columns = [c for c in ALL_FEATURES + [TARGET_COLUMN] if c not in df]
    if missing_columns:
        raise ValueError(f"{path} is missing required Covertype columns: {missing_columns}")
    if n_samples < 1 or n_samples > len(df):
        raise ValueError(f"Requested {n_samples:,} rows, but dataset has {len(df):,}.")
    df = df[ALL_FEATURES + [TARGET_COLUMN]]
    if df[TARGET_COLUMN].nunique() != 7:
        raise ValueError("Expected the seven Covertype target classes in Cover_Type.")
    source_rows = len(df)
    if n_samples == len(df):
        sampled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        sampled, _ = train_test_split(
            df, train_size=n_samples, stratify=df[TARGET_COLUMN],
            random_state=random_state,
        )
        sampled = sampled.reset_index(drop=True)
    sampled.attrs["source_rows"] = source_rows
    return sampled


def get_dataset_overview(df):
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    return {
        "shape": tuple(df.shape),
        "source_rows": int(df.attrs.get("source_rows", len(df))),
        "class_distribution": {
            "counts": counts.to_dict(),
            "percentages": (counts / len(df) * 100).round(3).to_dict(),
        },
        "feature_types": {
            "continuous": [f for f in CONTINUOUS_FEATURES if f in df],
            "binary": [f for f in BINARY_FEATURES if f in df],
        },
        "missing_values": int(df.isna().sum().sum()),
        "summary_statistics": df[ALL_FEATURES].describe().round(3),
    }


def split_data(df, test_size=0.15, val_size=0.15, random_state=RANDOM_STATE):
    """Return numpy arrays from a stratified 70/15/15 train/val/test split."""
    X = df[ALL_FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy()
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val, stratify=y_train_val,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, list(ALL_FEATURES)


def stratified_subsample(X, y, n, random_state=RANDOM_STATE):
    """Select exactly n stratified rows from array-like data."""
    if n >= len(y):
        return X, y
    indices = np.arange(len(y))
    selected, _ = train_test_split(
        indices, train_size=n, stratify=np.asarray(y), random_state=random_state,
    )
    return X[selected], np.asarray(y)[selected]


def get_plotting_subsample(df, n=5_000, random_state=RANDOM_STATE):
    """Stratified DataFrame sample used for pair plots."""
    if n >= len(df):
        return df.copy()
    sampled, _ = train_test_split(
        df, train_size=n, stratify=df[TARGET_COLUMN], random_state=random_state,
    )
    return sampled.reset_index(drop=True)


def get_decision_boundary_data(df, n=10_000, random_state=RANDOM_STATE):
    sample = get_plotting_subsample(df, n=n, random_state=random_state)
    return (
        sample[DECISION_BOUNDARY_FEATURES].to_numpy(dtype=np.float32),
        sample[TARGET_COLUMN].to_numpy(),
    )
