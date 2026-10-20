"""Train a compatibility classifier from the vendor CSV files.

Run ``py train_compatibility_model.py`` from the project directory, or pass a
CSV file/directory with ``--input``. The default target is the sourced ``zos32``
compatibility status; it is not inferred from an availability feature.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_CANDIDATES = [
    "vendor",
    "product",
    "vrm",
    "zos31",
    "zos25",
    "ptf_required",
]


def load_data(source: Path) -> pd.DataFrame:
    if source.is_dir():
        files = sorted(source.glob("*_compat.csv"))
    elif source.is_file():
        files = [source]
    else:
        raise FileNotFoundError(f"Input path does not exist: {source}")

    if not files:
        raise FileNotFoundError(
            f"No CSV files found at {source}. A directory must contain "
            "*_compat.csv files."
        )

    frames = [
        pd.read_csv(file, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for file in files
    ]
    return pd.concat(frames, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train and evaluate a z/OS compatibility status classifier."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "data",
        help="CSV file or directory of *_compat.csv files (default: ./data)",
    )
    parser.add_argument(
        "--target",
        default="zos32",
        help="Compatibility status column to predict (default: zos32)",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("compatibility_model.pkl"),
        help="Path for the saved scikit-learn pipeline",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of rows reserved for evaluation (default: 0.3)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1")

    df = load_data(args.input)
    required_columns = {"vendor", "product", args.target}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required column(s): "
            + ", ".join(missing_columns)
        )

    feature_columns = [
        column
        for column in FEATURE_CANDIDATES
        if column in df.columns and column != args.target
    ]
    if not {"vendor", "product"}.issubset(feature_columns):
        raise ValueError("Input CSV must contain both 'vendor' and 'product'.")

    df = df[feature_columns + [args.target]].copy()
    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()
    df = df[
        (df["vendor"] != "")
        & (df["product"] != "")
        & (df[args.target] != "")
    ]
    if len(df) < 2:
        raise ValueError("At least two rows with a known target are required.")
    if df[args.target].nunique() < 2:
        raise ValueError("At least two distinct target classes are required.")

    X = df[feature_columns]
    y = df[args.target]
    class_counts = y.value_counts()
    test_count = math.ceil(len(df) * args.test_size)
    train_count = len(df) - test_count
    can_stratify = (
        class_counts.min() >= 2
        and test_count >= len(class_counts)
        and train_count >= len(class_counts)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), feature_columns)
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(n_estimators=100, random_state=42),
            ),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Rows used: {len(df)}")
    print(f"Target: {args.target}")
    print("Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    print(f"Model pipeline saved to: {args.model_output}")


if __name__ == "__main__":
    main()
