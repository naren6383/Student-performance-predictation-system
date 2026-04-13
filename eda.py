"""
eda.py
-------
Step 3 — Exploratory Data Analysis (EDA)

What this script does:
  📊 Draws distribution plots for all key numeric features
  📉 Scatter plots: study hours vs final marks, attendance vs marks
  🔥 Correlation heatmap to find relationships between variables
  📦 Box plots: pass/fail distribution across features
  💡 Prints key insights discovered from the data

Simple explanation:
  Before we train any model, we LOOK at the data using charts.
  This helps us understand which features matter the most and
  whether the data has patterns our model can learn from.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (saves files)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(__file__))
DATA_PATH= os.path.join(BASE, "data", "student_cleaned.csv")
PLOTS_DIR= os.path.join(BASE, "outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
PALETTE = "coolwarm"
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def save(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"   💾  Saved → {path}")


def run_eda(data_path: str = DATA_PATH):
    df = pd.read_csv(data_path)
    print("=" * 55)
    print("📊  Exploratory Data Analysis (EDA)")
    print("=" * 55)

    # ── 1. Feature Distributions ──────────────────────────────────────────
    print("\n[1] Distribution plots …")
    num_cols = ["study_hours", "attendance_pct", "prev_marks",
                "sleep_hours", "final_marks"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Distribution of Key Features", fontsize=16, fontweight="bold")
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], ax=axes[i], kde=True,
                     color=COLORS[i % len(COLORS)], bins=25)
        axes[i].set_title(f"{col.replace('_', ' ').title()}")
        axes[i].set_xlabel("")
    axes[-1].axis("off")
    save(fig, "1_feature_distributions.png")

    # ── 2. Study Hours vs Final Marks ─────────────────────────────────────
    print("[2] Study hours vs Final marks …")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="study_hours", y="final_marks",
                    hue="pass_fail", palette={1: "#55A868", 0: "#C44E52"},
                    alpha=0.7, s=60, ax=ax)
    # regression line
    z = np.polyfit(df["study_hours"], df["final_marks"], 1)
    p = np.poly1d(z)
    xs = np.linspace(df["study_hours"].min(), df["study_hours"].max(), 200)
    ax.plot(xs, p(xs), "k--", linewidth=1.5, label="Trend Line")
    ax.set_title("Study Hours vs Final Marks", fontsize=14, fontweight="bold")
    ax.set_xlabel("Study Hours (normalised)")
    ax.set_ylabel("Final Marks")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Fail", "Pass", "Trend"], title="Result")
    save(fig, "2_study_hours_vs_marks.png")

    # ── 3. Attendance vs Final Marks ──────────────────────────────────────
    print("[3] Attendance vs Final marks …")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="attendance_pct", y="final_marks",
                    hue="pass_fail", palette={1: "#4C72B0", 0: "#DD8452"},
                    alpha=0.7, s=60, ax=ax)
    z = np.polyfit(df["attendance_pct"], df["final_marks"], 1)
    p = np.poly1d(z)
    xs = np.linspace(df["attendance_pct"].min(), df["attendance_pct"].max(), 200)
    ax.plot(xs, p(xs), "k--", linewidth=1.5, label="Trend Line")
    ax.set_title("Attendance % vs Final Marks", fontsize=14, fontweight="bold")
    ax.set_xlabel("Attendance % (normalised)")
    ax.set_ylabel("Final Marks")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Fail", "Pass", "Trend"], title="Result")
    save(fig, "3_attendance_vs_marks.png")

    # ── 4. Correlation Heatmap ────────────────────────────────────────────
    print("[4] Correlation heatmap …")
    fig, ax = plt.subplots(figsize=(10, 7))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0,
                linewidths=0.5, annot_kws={"size": 9}, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    save(fig, "4_correlation_heatmap.png")

    # ── 5. Box Plot: Pass / Fail across features ──────────────────────────
    print("[5] Box plots (Pass vs Fail) …")
    features = ["study_hours", "attendance_pct", "prev_marks", "sleep_hours"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Feature Distribution by Pass / Fail", fontsize=15, fontweight="bold")
    axes = axes.flatten()
    for i, feat in enumerate(features):
        sns.boxplot(data=df, x="pass_fail", y=feat,
                    palette={1: "#55A868", 0: "#C44E52"}, ax=axes[i])
        axes[i].set_title(feat.replace("_", " ").title())
        axes[i].set_xticklabels(["Fail (0)", "Pass (1)"])
        axes[i].set_xlabel("")
    save(fig, "5_boxplots_pass_fail.png")

    # ── 6. Pair Plot (quick overview) ─────────────────────────────────────
    print("[6] Pair plot …")
    pair_cols = ["study_hours", "attendance_pct", "prev_marks", "final_marks", "pass_fail"]
    pair_df = df[pair_cols].copy()
    pg = sns.pairplot(pair_df, hue="pass_fail",
                      palette={1: "#4C72B0", 0: "#C44E52"},
                      plot_kws={"alpha": 0.5, "s": 20},
                      diag_kind="kde")
    pg.figure.suptitle("Pair Plot of Key Features", y=1.02, fontsize=14, fontweight="bold")
    save(pg.figure, "6_pair_plot.png")

    # ── 7. Pass / Fail Count ──────────────────────────────────────────────
    print("[7] Pass/Fail count bar …")
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["pass_fail"].value_counts()
    bars = ax.bar(["Fail", "Pass"], counts.values,
                  color=["#C44E52", "#55A868"], edgecolor="white",
                  linewidth=1.2, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3, str(val),
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("Pass / Fail Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Students")
    save(fig, "7_pass_fail_count.png")

    # ── Print Insights ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("💡  KEY INSIGHTS FROM EDA")
    print("=" * 55)
    corr_final = df.corr(numeric_only=True)["final_marks"].drop("final_marks").sort_values(ascending=False)
    print(f"\n  Top feature correlations with Final Marks:")
    for feat, val in corr_final.items():
        bar = "█" * int(abs(val) * 20)
        print(f"    {feat:20s}  {val:+.3f}  {bar}")

    pass_rate = df["pass_fail"].mean() * 100
    print(f"\n  Overall pass rate    : {pass_rate:.1f}%")
    print(f"  Avg final marks      : {df['final_marks'].mean():.1f}")
    print(f"  Avg study hours (norm): {df['study_hours'].mean():.3f}")
    print(f"  Avg attendance (norm) : {df['attendance_pct'].mean():.3f}")
    print(f"\n  → More study hours → higher final marks (positive correlation)")
    print(f"  → Higher attendance → better performance")
    print(f"  → Previous marks is the strongest single predictor")
    print(f"\n✅  All plots saved to: {PLOTS_DIR}\n")


if __name__ == "__main__":
    run_eda()
