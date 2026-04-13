"""
main.py
--------
🚀 ENTRY POINT — Student Performance Prediction System

Run this single file to:
  Step 1 → Generate synthetic student dataset (500 students)
  Step 2 → Data preprocessing (missing values, encoding, normalisation)
  Step 3 → Exploratory Data Analysis (7 charts saved to outputs/plots/)
  Step 4 → Build & train Linear Regression (predict final marks)
  Step 5 → Build & train Logistic Regression (predict pass/fail)
  Step 6 → Show evaluation metrics (MAE, R², Accuracy, F1, Confusion Matrix)
  Step 7 → Interactive prediction from user input

Usage:
  python main.py
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import io
import pickle
import warnings

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Save plots to files (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing  import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model   import LinearRegression, LogisticRegression
from sklearn.metrics        import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  PATHS  (all relative to this file's directory)
# ──────────────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(ROOT, "data")
PLOTS_DIR = os.path.join(ROOT, "outputs", "plots")
MODEL_DIR = os.path.join(ROOT, "outputs", "models")
RAW_CSV   = os.path.join(DATA_DIR,  "student_data.csv")
CLEAN_CSV = os.path.join(DATA_DIR,  "student_cleaned.csv")

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

def header(title: str, step: str = ""):
    tag = f"[{step}] " if step else ""
    print(f"\n{'═'*60}")
    print(f"  {tag}{title}")
    print(f"{'═'*60}\n")

def save_fig(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"   💾  Plot saved → {path}")

def save_model(obj, name: str):
    path = os.path.join(MODEL_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"   💾  Model saved → {path}")

def load_model(name: str):
    path = os.path.join(MODEL_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)

# ──────────────────────────────────────────────────────────────────────────────
#  STEP 1 — GENERATE DATASET
# ──────────────────────────────────────────────────────────────────────────────
def step1_generate_dataset(n: int = 500):
    """
    Creates a realistic synthetic dataset of n students.

    Features:
      study_hours     — hours studied per day (0.5 – 10)
      attendance_pct  — attendance percentage (50 – 100)
      prev_marks      — previous semester marks (30 – 100)
      sleep_hours     — average sleep hours per night (4 – 10)
      extracurricular — joins extra activities? Yes / No
      internet_access — has internet at home? Yes / No
      gender          — Male / Female

    Targets:
      final_marks — exam score (0–100, influenced by above features + noise)
      pass_fail   — 1 if final_marks ≥ 40 else 0
    """
    header("Generating Dataset", "STEP 1")
    np.random.seed(42)

    gender     = np.random.choice(["Male", "Female"], n, p=[0.52, 0.48])
    study_h    = np.round(np.random.uniform(0.5, 10,  n), 1)
    attend     = np.round(np.random.uniform(50,  100, n), 1)
    prev_m     = np.round(np.random.uniform(30,  100, n), 1)
    sleep_h    = np.round(np.random.uniform(4,   10,  n), 1)
    extra      = np.random.choice(["Yes", "No"],  n, p=[0.40, 0.60])
    internet   = np.random.choice(["Yes", "No"],  n, p=[0.70, 0.30])

    noise      = np.random.normal(0, 5, n)
    final_m    = (
        0.35 * study_h * 5     +
        0.25 * attend  * 0.6   +
        0.30 * prev_m  * 0.7   +
        0.05 * sleep_h * 2     +
        noise
    )
    final_m    = np.clip(np.round(final_m, 1), 0, 100)
    pass_fail  = (final_m >= 40).astype(int)

    def inject_nulls(arr, pct=0.03):
        arr  = arr.astype(object)
        idx  = np.random.choice(len(arr), size=int(len(arr)*pct), replace=False)
        arr[idx] = np.nan
        return arr

    df = pd.DataFrame({
        "student_id"     : [f"S{str(i).zfill(4)}" for i in range(1, n+1)],
        "gender"         : gender,
        "study_hours"    : inject_nulls(study_h),
        "attendance_pct" : inject_nulls(attend),
        "prev_marks"     : inject_nulls(prev_m),
        "sleep_hours"    : inject_nulls(sleep_h),
        "extracurricular": extra,
        "internet_access": internet,
        "final_marks"    : final_m,
        "pass_fail"      : pass_fail,
    })

    df.to_csv(RAW_CSV, index=False)
    print(f"  ✅  Dataset ({n} rows × {df.shape[1]} cols)  →  {RAW_CSV}")
    print(f"  Pass rate : {pass_fail.mean()*100:.1f}%   |"
          f"  Avg marks : {final_m.mean():.1f}\n")
    print(df.head(5).to_string(index=False))
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  STEP 2 — PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "study_hours", "attendance_pct", "prev_marks",
    "sleep_hours", "gender", "extracurricular", "internet_access"
]

def step2_preprocess():
    """
    Preprocessing pipeline:
      1. Load raw CSV
      2. Inspect missing values
      3. Fill numerics with median, categoricals with mode
      4. Label-encode gender / extracurricular / internet_access
      5. Min-Max normalise all feature columns
      6. Save cleaned CSV + return scaler for later use
    """
    header("Data Preprocessing", "STEP 2")
    df = pd.read_csv(RAW_CSV)

    print("  📋  Shape       :", df.shape)
    print("  📋  Missing vals:\n", df.isnull().sum(), "\n")

    # — Fill missing values —
    num_fill = ["study_hours", "attendance_pct", "prev_marks", "sleep_hours"]
    for c in num_fill:
        med = df[c].median()
        n   = df[c].isnull().sum()
        if n:
            df[c] = df[c].fillna(med)
            print(f"  Filled {n:>3} NaN in {c:20s} (median={med:.2f})")

    # — Encode categoricals —
    mapping = {
        "gender"         : {"Male": 0, "Female": 1},
        "extracurricular": {"Yes": 1, "No": 0},
        "internet_access": {"Yes": 1, "No": 0},
    }
    for col, m in mapping.items():
        df[col] = df[col].map(m)
    print("\n  Encoding done:", list(mapping.keys()))

    # — Normalise —
    scaler = MinMaxScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    print("  Normalisation (Min-Max) applied to:", FEATURE_COLS)

    # Drop identifier
    df.drop(columns=["student_id"], inplace=True, errors="ignore")
    df.to_csv(CLEAN_CSV, index=False)
    print(f"\n  ✅  Cleaned CSV → {CLEAN_CSV}\n")
    print(df.head(3).to_string(index=False))
    return df, scaler


# ──────────────────────────────────────────────────────────────────────────────
#  STEP 3 — EDA
# ──────────────────────────────────────────────────────────────────────────────
def step3_eda(df: pd.DataFrame):
    """
    7 Visualisations:
      1. Feature distributions (histograms + KDE)
      2. Study hours vs final marks  (scatter + trend)
      3. Attendance % vs final marks (scatter + trend)
      4. Correlation heatmap
      5. Box plots: pass/fail across features
      6. Pair plot
      7. Pass / Fail count bar
    """
    header("Exploratory Data Analysis", "STEP 3")

    # 1 — Distributions
    print("  [1] Feature distributions …")
    num_cols = ["study_hours", "attendance_pct", "prev_marks", "sleep_hours", "final_marks"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold")
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], ax=axes[i], kde=True, color=COLORS[i], bins=25)
        axes[i].set_title(col.replace("_", " ").title())
    axes[-1].axis("off")
    save_fig(fig, "01_distributions.png")

    # 2 — Study hours vs Marks
    print("  [2] Study hours vs Final marks …")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="study_hours", y="final_marks",
                    hue="pass_fail",
                    palette={1: "#55A868", 0: "#C44E52"},
                    alpha=0.65, s=55, ax=ax)
    xs = np.linspace(df["study_hours"].min(), df["study_hours"].max(), 200)
    z  = np.polyfit(df["study_hours"], df["final_marks"], 1)
    ax.plot(xs, np.poly1d(z)(xs), "k--", linewidth=1.5, label="Trend")
    ax.set_title("Study Hours vs Final Marks", fontsize=14, fontweight="bold")
    ax.set_xlabel("Study Hours (normalised)")
    ax.set_ylabel("Final Marks")
    ax.legend(title="Outcome", labels=["Fail", "Pass", "Trend"])
    save_fig(fig, "02_study_hours_vs_marks.png")

    # 3 — Attendance vs Marks
    print("  [3] Attendance vs Final marks …")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="attendance_pct", y="final_marks",
                    hue="pass_fail",
                    palette={1: "#4C72B0", 0: "#DD8452"},
                    alpha=0.65, s=55, ax=ax)
    xs = np.linspace(df["attendance_pct"].min(), df["attendance_pct"].max(), 200)
    z  = np.polyfit(df["attendance_pct"], df["final_marks"], 1)
    ax.plot(xs, np.poly1d(z)(xs), "k--", linewidth=1.5, label="Trend")
    ax.set_title("Attendance % vs Final Marks", fontsize=14, fontweight="bold")
    ax.set_xlabel("Attendance % (normalised)")
    ax.set_ylabel("Final Marks")
    ax.legend(title="Outcome", labels=["Fail", "Pass", "Trend"])
    save_fig(fig, "03_attendance_vs_marks.png")

    # 4 — Correlation heatmap
    print("  [4] Correlation heatmap …")
    fig, ax = plt.subplots(figsize=(10, 7))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0,
                linewidths=0.5, annot_kws={"size": 9}, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    save_fig(fig, "04_correlation_heatmap.png")

    # 5 — Box plots
    print("  [5] Box plots (Pass vs Fail) …")
    feats = ["study_hours", "attendance_pct", "prev_marks", "sleep_hours"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Feature Distribution by Pass / Fail", fontsize=15, fontweight="bold")
    axes = axes.flatten()
    for i, feat in enumerate(feats):
        sns.boxplot(data=df, x="pass_fail", y=feat, ax=axes[i])
        axes[i].set_title(feat.replace("_", " ").title())
        axes[i].set_xlabel("")
    save_fig(fig, "05_boxplots_pass_fail.png")

    # 6 — Pair plot
    print("  [6] Pair plot …")
    pair_cols = ["study_hours", "attendance_pct", "prev_marks", "final_marks", "pass_fail"]
    pg = sns.pairplot(df[pair_cols], hue="pass_fail",
                      palette={1: "#4C72B0", 0: "#C44E52"},
                      plot_kws={"alpha": 0.45, "s": 18}, diag_kind="kde")
    pg.figure.suptitle("Pair Plot", y=1.02, fontsize=14, fontweight="bold")
    save_fig(pg.figure, "06_pair_plot.png")

    # 7 — Pass/Fail count
    print("  [7] Pass/Fail count …")
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["pass_fail"].value_counts()
    bars   = ax.bar(["Fail", "Pass"], counts.values,
                    color=["#C44E52", "#55A868"],
                    edgecolor="white", linewidth=1.2, width=0.5)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 2, str(v),
                ha="center", fontweight="bold")
    ax.set_title("Pass / Fail Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Students")
    save_fig(fig, "07_pass_fail_count.png")

    # — Insights —
    print("\n  💡  KEY INSIGHTS:")
    corr_f = df.corr(numeric_only=True)["final_marks"].drop("final_marks").sort_values(ascending=False)
    for feat, val in corr_f.items():
        bar = "█" * int(abs(val) * 25)
        print(f"    {feat:22s}  r={val:+.3f}  {bar}")
    print(f"\n  Pass rate : {df['pass_fail'].mean()*100:.1f}%")
    print(f"  Avg marks : {df['final_marks'].mean():.1f}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  STEP 4-6 — MACHINE LEARNING MODELS
# ──────────────────────────────────────────────────────────────────────────────
def step4_train_models(df: pd.DataFrame):
    """
    LINEAR REGRESSION   → predicts continuous final_marks score
    LOGISTIC REGRESSION → predicts binary pass/fail class
    """
    header("Training & Evaluating ML Models", "STEP 4-6")

    X     = df[FEATURE_COLS]
    y_reg = df["final_marks"]
    y_clf = df["pass_fail"]

    X_tr, X_te, yr_tr, yr_te = train_test_split(
        X, y_reg, test_size=0.2, random_state=42)
    _,    _,    yc_tr, yc_te = train_test_split(
        X, y_clf, test_size=0.2, random_state=42)

    print(f"  Train / Test split : {len(X_tr)} / {len(X_te)} samples\n")

    # ── LINEAR REGRESSION ─────────────────────────────────────────────────
    print("  📈  LINEAR REGRESSION (Final Marks Prediction)")
    print("  " + "─"*50)
    lr = LinearRegression()
    lr.fit(X_tr, yr_tr)
    yr_pred = lr.predict(X_te)

    mae  = mean_absolute_error(yr_te, yr_pred)
    rmse = np.sqrt(mean_squared_error(yr_te, yr_pred))
    r2   = r2_score(yr_te, yr_pred)

    print(f"  MAE  (avg error)         : {mae:.2f} marks")
    print(f"  RMSE (penalised error)   : {rmse:.2f} marks")
    print(f"  R²   (explained variance): {r2:.4f}  ({r2*100:.1f}%)\n")

    coef_s = pd.Series(lr.coef_, index=FEATURE_COLS).sort_values(ascending=False)
    print("  Feature Coefficients (impact on predicted marks):")
    for f, v in coef_s.items():
        bar = "█" * int(abs(v) / 3)
        print(f"    {f:22s}  {v:+.2f}  {bar}")
    print()

    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(yr_te, yr_pred, alpha=0.55, color="#4C72B0", s=35)
    mn, mx = yr_te.min(), yr_te.max()
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect")
    ax.set_xlabel("Actual Final Marks")
    ax.set_ylabel("Predicted Final Marks")
    ax.set_title("Linear Regression — Actual vs Predicted", fontsize=13, fontweight="bold")
    ax.legend()
    save_fig(fig, "08_linreg_actual_vs_pred.png")

    # Residuals
    resid = yr_te.values - yr_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(yr_pred, resid, alpha=0.5, color="#DD8452", s=35)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Predicted Marks")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot", fontsize=13, fontweight="bold")
    save_fig(fig, "09_linreg_residuals.png")

    # Coefficient bar
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#55A868" if v > 0 else "#C44E52" for v in coef_s.values]
    ax.barh(coef_s.index, coef_s.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Linear Regression — Feature Coefficients", fontsize=13, fontweight="bold")
    save_fig(fig, "10_linreg_coefficients.png")

    save_model(lr, "linear_regression.pkl")

    # ── LOGISTIC REGRESSION ───────────────────────────────────────────────
    print("\n  🎯  LOGISTIC REGRESSION (Pass / Fail Prediction)")
    print("  " + "─"*50)
    lc = LogisticRegression(max_iter=1000, random_state=42)
    lc.fit(X_tr, yc_tr)
    yc_pred = lc.predict(X_te)
    yc_prob = lc.predict_proba(X_te)[:, 1]

    acc  = accuracy_score (yc_te, yc_pred)
    prec = precision_score(yc_te, yc_pred, zero_division=0)
    rec  = recall_score   (yc_te, yc_pred, zero_division=0)
    f1   = f1_score       (yc_te, yc_pred, zero_division=0)
    cm   = confusion_matrix(yc_te, yc_pred)

    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision  : {prec:.4f}  ({prec*100:.1f}%)")
    print(f"  Recall     : {rec:.4f}  ({rec*100:.1f}%)")
    print(f"  F1-Score   : {f1:.4f}  ({f1*100:.1f}%)\n")
    print(classification_report(yc_te, yc_pred,
                                target_names=["Fail (0)", "Pass (1)"]))

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Fail", "Pass"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Logistic Regression", fontsize=13, fontweight="bold")
    save_fig(fig, "11_logreg_confusion_matrix.png")

    # Probability distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(yc_prob[yc_te == 0], bins=25, alpha=0.65, color="#C44E52", label="Actual: Fail")
    ax.hist(yc_prob[yc_te == 1], bins=25, alpha=0.65, color="#55A868", label="Actual: Pass")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Decision Boundary")
    ax.set_title("Predicted Probability of Passing", fontsize=13, fontweight="bold")
    ax.set_xlabel("P(Pass)")
    ax.set_ylabel("Count")
    ax.legend()
    save_fig(fig, "12_logreg_probability.png")

    # Model comparison summary
    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║                  MODEL SUMMARY                        ║
  ╠══════════════════════════╦═══════════════════════════╣
  ║  Linear Regression        ║  R²       = {r2:.4f}       ║
  ║  (Final Marks)            ║  MAE      = {mae:.2f} marks  ║
  ╠══════════════════════════╬═══════════════════════════╣
  ║  Logistic Regression      ║  Accuracy = {acc*100:.1f}%        ║
  ║  (Pass / Fail)            ║  F1-Score = {f1:.4f}       ║
  ╚══════════════════════════╩═══════════════════════════╝
    """)

    save_model(lc, "logistic_regression.pkl")
    return lr, lc


# ──────────────────────────────────────────────────────────────────────────────
#  STEP 7 — INTERACTIVE PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
def _get_float(prompt, lo, hi):
    while True:
        try:
            v = float(input(prompt))
            if lo <= v <= hi:
                return v
            print(f"  ⚠  Enter a value between {lo} and {hi}.")
        except ValueError:
            print("  ⚠  Enter a valid number.")

def _get_choice(prompt, choices):
    while True:
        v = input(prompt).strip().title()
        if v in choices:
            return v
        print(f"  ⚠  Choose from: {choices}")

def _normalise_input(raw: dict, df_clean: pd.DataFrame) -> np.ndarray:
    """Apply same min-max scaling using the cleaned training distribution."""
    norm = []
    for col in FEATURE_COLS:
        lo = df_clean[col].min()
        hi = df_clean[col].max()
        v  = raw[col]
        norm.append((v - lo) / (hi - lo) if hi != lo else 0.0)
    return np.array(norm).reshape(1, -1)

def step7_predict(lr, lc):
    """
    Interactive loop: ask user for details, predict marks + pass/fail,
    show confidence bar, give personalised recommendations.
    """
    header("Interactive Student Prediction", "STEP 7")
    df_clean = pd.read_csv(CLEAN_CSV)   # needed for normalisation ranges

    while True:
        print("  📝  Enter student details:\n")
        study_h  = _get_float("   Daily study hours      (0.5 – 10.0) : ", 0.5, 10.0)
        attend   = _get_float("   Attendance %            (50 – 100)   : ", 50.0, 100.0)
        prev_m   = _get_float("   Previous semester marks (30 – 100)   : ", 30.0, 100.0)
        sleep_h  = _get_float("   Sleep hours per night   (4 – 10)     : ", 4.0, 10.0)
        gender   = _get_choice("   Gender                  (Male/Female): ", ["Male", "Female"])
        extra    = _get_choice("   Extracurricular         (Yes/No)     : ", ["Yes", "No"])
        internet = _get_choice("   Internet at home        (Yes/No)     : ", ["Yes", "No"])

        raw = {
            "study_hours"    : study_h,
            "attendance_pct" : attend,
            "prev_marks"     : prev_m,
            "sleep_hours"    : sleep_h,
            "gender"         : 1.0 if gender   == "Female" else 0.0,
            "extracurricular": 1.0 if extra    == "Yes"    else 0.0,
            "internet_access": 1.0 if internet == "Yes"    else 0.0,
        }
        feat_vec = _normalise_input(raw, df_clean)

        pred_marks   = float(np.clip(lr.predict(feat_vec)[0], 0, 100))
        pass_prob    = float(lc.predict_proba(feat_vec)[0][1])
        pred_outcome = "✅ PASS" if pass_prob >= 0.5 else "❌ FAIL"

        print("\n" + "═"*55)
        print("  🎯  PREDICTION RESULTS")
        print("═"*55)
        print(f"  Predicted Final Marks : {pred_marks:.1f} / 100")
        print(f"  Pass Probability      : {pass_prob*100:.1f}%")
        print(f"  Expected Outcome      : {pred_outcome}")

        bar_len = int(pass_prob * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"\n  Pass Confidence [{bar}] {pass_prob*100:.0f}%")

        # Personalised tips
        print("\n  📚  Recommendations:")
        if pred_marks < 40:
            tips = [
                "⚠  High fail risk! Study more hours each day.",
                "📅  Never skip class — attendance is critical.",
                "😴  Sleep 7-8 hours to improve memory retention.",
                "🧑‍🏫  Seek teacher or peer support immediately.",
            ]
        elif pred_marks < 60:
            tips = [
                "📈  Borderline pass. Add 1-2 study hours daily.",
                "🗒  Revise previous semester topics.",
                "📶  Use free resources: Khan Academy, YouTube.",
            ]
        elif pred_marks < 80:
            tips = [
                "👍  Good track record. Stay consistent!",
                "🎯  Nail weak topics for even better marks.",
                "📝  Practice past exam papers.",
            ]
        else:
            tips = [
                "🏆  Outstanding! You're on track for distinction.",
                "🚀  Explore advanced topics or mentor classmates.",
                "🎓  Keep those habits — consistency wins.",
            ]
        for t in tips:
            print(f"      {t}")

        print("\n" + "─"*55)
        again = input("  Predict for another student? (yes / no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\n  🎓  Thank you for using Student Performance Prediction System!\n")
            break


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════╗
║       🎓  STUDENT PERFORMANCE PREDICTION SYSTEM           ║
║            Final Year Data Science Project (FDS)          ║
╚══════════════════════════════════════════════════════════╝
"""

def main():
    print(BANNER)
    args = sys.argv[1:]

    # Step 1
    if "--skip-data" not in args:
        step1_generate_dataset(n=500)
    else:
        print("⏭  Skipping data generation.")

    # Step 2
    df_clean, scaler = step2_preprocess()

    # Step 3
    if "--skip-eda" not in args:
        step3_eda(df_clean)
    else:
        print("⏭  Skipping EDA.")

    # Step 4-6
    lr, lc = step4_train_models(df_clean)

    # Step 7
    if "--no-predict" not in args:
        step7_predict(lr, lc)
    else:
        print("\n✅  Pipeline complete. Run with no flags to enable interactive prediction.\n")


if __name__ == "__main__":
    main()
