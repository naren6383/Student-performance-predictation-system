"""
models.py
----------
Step 4–6 — Build, Train, Evaluate ML Models

Models built:
  1. LINEAR REGRESSION  →  Predict final_marks (a number)
  2. LOGISTIC REGRESSION →  Predict pass_fail  (0 or 1)

Simple explanation:
  • Linear Regression   : "Draw the best straight line through the data"
                          so we can predict a score like 72.5.
  • Logistic Regression : "Find the boundary between two classes"
                          so we can say Pass or Fail.

  Evaluation metrics explained:
    Regression:
      MAE  – Average distance between predicted & actual marks
      RMSE – Similar but penalises big errors more
      R²   – How much of the variation our model explains (1.0 = perfect)

    Classification:
      Accuracy   – How often the model is correct overall
      Precision  – Of all predicted "Pass", how many truly passed?
      Recall     – Of all real "Pass", how many did we catch?
      F1-Score   – Harmonic mean of precision & recall (balance)
      Confusion Matrix – Shows True Positives, False Positives, etc.
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model   import LinearRegression, LogisticRegression
from sklearn.metrics        import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE, "data",    "student_cleaned.csv")
PLOTS_DIR = os.path.join(BASE, "outputs", "plots")
MODEL_DIR = os.path.join(BASE, "outputs", "models")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"   💾  Plot → {path}")


def save_model(model, name):
    path = os.path.join(MODEL_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"   💾  Model → {path}")


# ── Feature / Target definition ────────────────────────────────────────────
FEATURES = [
    "study_hours", "attendance_pct", "prev_marks",
    "sleep_hours", "gender", "extracurricular", "internet_access"
]

def prepare_data(data_path: str = DATA_PATH, test_size: float = 0.2):
    df = pd.read_csv(data_path)
    X = df[FEATURES]

    y_reg = df["final_marks"]    # regression target
    y_clf = df["pass_fail"]      # classification target

    X_tr, X_te, yr_tr, yr_te = train_test_split(
        X, y_reg, test_size=test_size, random_state=42
    )
    _, _, yc_tr, yc_te = train_test_split(
        X, y_clf, test_size=test_size, random_state=42
    )

    print(f"  Training samples  : {len(X_tr)}")
    print(f"  Testing  samples  : {len(X_te)}")
    print(f"  Features used     : {FEATURES}\n")
    return X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te


# ────────────────────────────────────────────────────────────────────────────
#  MODEL 1 : LINEAR REGRESSION (predict final marks)
# ────────────────────────────────────────────────────────────────────────────
def train_linear_regression(X_tr, X_te, yr_tr, yr_te):
    print("=" * 55)
    print("📈  MODEL 1 — Linear Regression (Final Marks)")
    print("=" * 55)

    lr = LinearRegression()
    lr.fit(X_tr, yr_tr)           # TRAINING
    yr_pred = lr.predict(X_te)    # PREDICTION

    # ── Metrics ──
    mae  = mean_absolute_error(yr_te, yr_pred)
    rmse = np.sqrt(mean_squared_error(yr_te, yr_pred))
    r2   = r2_score(yr_te, yr_pred)

    print(f"\n  📊  Evaluation Metrics:")
    print(f"    MAE  (Avg Error)        : {mae:.2f} marks")
    print(f"    RMSE (Penalised Error)  : {rmse:.2f} marks")
    print(f"    R²   (Variance Explained): {r2:.4f}  ({r2*100:.1f}%)")

    # Feature importance (coefficients)
    coef_df = pd.Series(lr.coef_, index=FEATURES).sort_values(ascending=False)
    print(f"\n  🏆  Feature Coefficients (impact on predicted marks):")
    for feat, val in coef_df.items():
        bar = "█" * int(abs(val) / 3)
        print(f"    {feat:20s}  {val:+.2f}  {bar}")

    # ── Actual vs Predicted scatter ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(yr_te, yr_pred, alpha=0.55, color="#4C72B0", s=40, label="Predictions")
    mn, mx = yr_te.min(), yr_te.max()
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual Final Marks")
    ax.set_ylabel("Predicted Final Marks")
    ax.set_title("Linear Regression — Actual vs Predicted Marks",
                 fontsize=13, fontweight="bold")
    ax.legend()
    save_fig(fig, "8_linear_regression_actual_vs_pred.png")

    # ── Residual Plot ──
    residuals = yr_te.values - yr_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(yr_pred, residuals, alpha=0.5, color="#DD8452", s=40)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Predicted Marks")
    ax.set_ylabel("Residuals (Actual − Predicted)")
    ax.set_title("Residual Plot (should scatter around 0)", fontsize=13, fontweight="bold")
    save_fig(fig, "9_linear_regression_residuals.png")

    # ── Coefficient Bar Chart ──
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#55A868" if v > 0 else "#C44E52" for v in coef_df.values]
    ax.barh(coef_df.index, coef_df.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Linear Regression — Feature Coefficients",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Coefficient Value")
    save_fig(fig, "10_lr_feature_coefficients.png")

    save_model(lr, "linear_regression.pkl")
    print()
    return lr, r2


# ────────────────────────────────────────────────────────────────────────────
#  MODEL 2 : LOGISTIC REGRESSION (predict pass / fail)
# ────────────────────────────────────────────────────────────────────────────
def train_logistic_regression(X_tr, X_te, yc_tr, yc_te):
    print("=" * 55)
    print("🎯  MODEL 2 — Logistic Regression (Pass / Fail)")
    print("=" * 55)

    lc = LogisticRegression(max_iter=1000, random_state=42)
    lc.fit(X_tr, yc_tr)           # TRAINING
    yc_pred = lc.predict(X_te)    # PREDICTION
    yc_prob = lc.predict_proba(X_te)[:, 1]   # probability of passing

    # ── Metrics ──
    acc  = accuracy_score (yc_te, yc_pred)
    prec = precision_score(yc_te, yc_pred, zero_division=0)
    rec  = recall_score   (yc_te, yc_pred, zero_division=0)
    f1   = f1_score       (yc_te, yc_pred, zero_division=0)
    cm   = confusion_matrix(yc_te, yc_pred)

    print(f"\n  📊  Evaluation Metrics:")
    print(f"    Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"    Precision  : {prec:.4f}  ({prec*100:.1f}%)")
    print(f"    Recall     : {rec:.4f}  ({rec*100:.1f}%)")
    print(f"    F1-Score   : {f1:.4f}  ({f1*100:.1f}%)")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Full Classification Report:\n")
    print(classification_report(yc_te, yc_pred,
                                target_names=["Fail (0)", "Pass (1)"]))

    # ── Confusion Matrix Plot ──
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Fail", "Pass"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Logistic Regression",
                 fontsize=13, fontweight="bold")
    save_fig(fig, "11_logistic_confusion_matrix.png")

    # ── Probability Distribution ──
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(yc_prob[yc_te == 0], bins=25, alpha=0.65,
            color="#C44E52", label="Actual: Fail")
    ax.hist(yc_prob[yc_te == 1], bins=25, alpha=0.65,
            color="#55A868", label="Actual: Pass")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Decision Boundary")
    ax.set_title("Predicted Probability of Passing",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("P(Pass)")
    ax.set_ylabel("Count")
    ax.legend()
    save_fig(fig, "12_logistic_probability_dist.png")

    # ── Coefficient importance ──
    coef_s = pd.Series(lc.coef_[0], index=FEATURES).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#55A868" if v > 0 else "#C44E52" for v in coef_s.values]
    ax.barh(coef_s.index, coef_s.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Logistic Regression — Feature Coefficients (log-odds)",
                 fontsize=13, fontweight="bold")
    save_fig(fig, "13_logistic_feature_coefficients.png")

    save_model(lc, "logistic_regression.pkl")
    print()
    return lc, acc


# ────────────────────────────────────────────────────────────────────────────
#  COMPARISON TABLE
# ────────────────────────────────────────────────────────────────────────────
def print_summary(r2: float, acc: float):
    print("=" * 55)
    print("📋  MODEL SUMMARY")
    print("=" * 55)
    print(f"""
  ┌──────────────────────────┬────────────────────┐
  │ Model                    │ Key Metric         │
  ├──────────────────────────┼────────────────────┤
  │ Linear Regression        │ R² = {r2:.4f}        │
  │ Logistic Regression      │ Acc= {acc*100:.1f}%           │
  └──────────────────────────┴────────────────────┘
    """)


def train_models():
    print("\n🚀  Starting Model Training …\n")
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = prepare_data()
    lr, r2  = train_linear_regression (X_tr, X_te, yr_tr, yr_te)
    lc, acc = train_logistic_regression(X_tr, X_te, yc_tr, yc_te)
    print_summary(r2, acc)
    return lr, lc


if __name__ == "__main__":
    train_models()
