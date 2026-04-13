"""
preprocessing.py
-----------------
Step 2 — Data Preprocessing

What this script does:
  ✅ Loads the raw CSV
  ✅ Inspects data types, shape, and missing values
  ✅ Fills missing numeric values with the column median
  ✅ Encodes categorical (text) columns into numbers
  ✅ Normalises numeric features using Min-Max scaling
  ✅ Saves the cleaned dataset for later use

Simple explanation:
  - Machines can only work with numbers.
  - Missing values are replaced so we don't lose rows.
  - "Male/Female" → 0/1, "Yes/No" → 1/0 (Label Encoding).
  - Normalising puts all numbers in the 0-1 range so large values
    don't dominate small ones during training.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(__file__))   # project root
RAW  = os.path.join(BASE, "data", "student_data.csv")
OUT  = os.path.join(BASE, "data", "student_cleaned.csv")

def load_and_inspect(path: str) -> pd.DataFrame:
    """Load CSV and print basic statistics."""
    df = pd.read_csv(path)
    print("=" * 55)
    print("📋  RAW DATASET OVERVIEW")
    print("=" * 55)
    print(f"  Shape     : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n  Data Types:\n{df.dtypes}\n")
    print(f"  Missing Values:\n{df.isnull().sum()}\n")
    print(f"  Descriptive Stats:\n{df.describe()}\n")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
      Numeric  → fill with column MEDIAN (robust to outliers)
      Categorical → fill with MODE (most frequent value)
    """
    print("🔧  Handling Missing Values …")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Exclude identifiers
    for col in ["student_id", "final_marks", "pass_fail"]:
        if col in num_cols: num_cols.remove(col)
        if col in cat_cols: cat_cols.remove(col)

    for col in num_cols:
        median = df[col].median()
        missing = df[col].isnull().sum()
        if missing:
            df[col] = df[col].fillna(median)
            print(f"   {col:20s} → filled {missing} NaN with median={median:.2f}")

    for col in cat_cols:
        mode = df[col].mode()[0]
        missing = df[col].isnull().sum()
        if missing:
            df[col] = df[col].fillna(mode)
            print(f"   {col:20s} → filled {missing} NaN with mode='{mode}'")

    print(f"\n  Remaining nulls: {df.isnull().sum().sum()}\n")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label Encoding:
      gender          : Male→0, Female→1
      extracurricular : Yes→1, No→0
      internet_access : Yes→1, No→0
    """
    print("🔤  Encoding Categorical Columns …")
    mappings = {
        "gender"         : {"Male": 0, "Female": 1},
        "extracurricular": {"Yes": 1, "No": 0},
        "internet_access": {"Yes": 1, "No": 0},
    }
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            print(f"   {col:20s} → {mapping}")
    print()
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-Max Normalisation on input features (not on target columns).
    Formula: x_scaled = (x - x_min) / (x_max - x_min)  →  [0, 1]
    """
    print("📐  Normalising Numeric Features (Min-Max) …")
    feature_cols = [
        "study_hours", "attendance_pct", "prev_marks",
        "sleep_hours", "gender", "extracurricular", "internet_access"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"   Scaled columns: {feature_cols}\n")
    return df, scaler


def preprocess(raw_path: str = RAW, out_path: str = OUT):
    df = load_and_inspect(raw_path)
    df = handle_missing(df)
    df = encode_categorical(df)
    df, scaler = normalize_features(df)

    # Drop student_id (not useful for training)
    df.drop(columns=["student_id"], inplace=True, errors="ignore")

    df.to_csv(out_path, index=False)
    print(f"✅  Cleaned dataset saved → {out_path}")
    print(df.head())
    return df, scaler


if __name__ == "__main__":
    preprocess()
