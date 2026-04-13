"""
generate_report.py
-------------------
Generates a beautiful self-contained HTML report after main.py has run.

Usage:
    python generate_report.py

Requirements:
    Run main.py first so that outputs/plots/ and outputs/models/ exist.
"""

import os
import base64
import pickle
import datetime
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(ROOT, "outputs", "plots")
MODEL_DIR = os.path.join(ROOT, "outputs", "models")
DATA_PATH = os.path.join(ROOT, "data", "student_cleaned.csv")
OUT_HTML  = os.path.join(ROOT, "outputs", "report.html")

os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)


def img_to_b64(path: str) -> str:
    """Convert an image file to a base64-encoded data URI."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def get_metrics():
    """Load models and compute metrics from cleaned CSV."""
    try:
        lr = pickle.load(open(os.path.join(MODEL_DIR, "linear_regression.pkl"),  "rb"))
        lc = pickle.load(open(os.path.join(MODEL_DIR, "logistic_regression.pkl"), "rb"))
    except FileNotFoundError:
        return None, None, None, None, None, None, None, None

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        mean_absolute_error, r2_score, mean_squared_error,
        accuracy_score, f1_score, precision_score, recall_score
    )

    df = pd.read_csv(DATA_PATH)
    FEATURES = ["study_hours", "attendance_pct", "prev_marks",
                 "sleep_hours", "gender", "extracurricular", "internet_access"]
    X  = df[FEATURES]
    yr = df["final_marks"]
    yc = df["pass_fail"]

    X_tr, X_te, yr_tr, yr_te = train_test_split(X, yr, test_size=0.2, random_state=42)
    _,    _,    yc_tr, yc_te = train_test_split(X, yc, test_size=0.2, random_state=42)

    yr_pred = lr.predict(X_te)
    yc_pred = lc.predict(X_te)

    mae  = mean_absolute_error(yr_te, yr_pred)
    rmse = float(np.sqrt(mean_squared_error(yr_te, yr_pred)))
    r2   = r2_score(yr_te, yr_pred)
    acc  = accuracy_score(yc_te, yc_pred)
    prec = precision_score(yc_te, yc_pred, zero_division=0)
    rec  = recall_score(yc_te, yc_pred, zero_division=0)
    f1   = f1_score(yc_te, yc_pred, zero_division=0)
    pass_rate = df["pass_fail"].mean() * 100

    return mae, rmse, r2, acc, prec, rec, f1, pass_rate


def card(title, value, subtitle="", color="#4C72B0"):
    return f"""
        <div class="metric-card" style="border-top: 4px solid {color};">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>"""


def build_plot_gallery():
    plot_map = [
        ("01_distributions.png",              "Feature Distributions",         "Histogram + KDE of each feature"),
        ("02_study_hours_vs_marks.png",        "Study Hours vs Final Marks",    "Shows the positive correlation"),
        ("03_attendance_vs_marks.png",         "Attendance vs Final Marks",     "Higher attendance → better marks"),
        ("04_correlation_heatmap.png",         "Correlation Heatmap",           "Strength of relationships between features"),
        ("05_boxplots_pass_fail.png",          "Box Plots: Pass vs Fail",       "Feature spread by outcome"),
        ("06_pair_plot.png",                   "Pair Plot",                     "All-feature pairwise scatter"),
        ("07_pass_fail_count.png",             "Pass / Fail Count",             "Class distribution"),
        ("08_linreg_actual_vs_pred.png",       "Actual vs Predicted (LR)",      "Linear Regression performance"),
        ("09_linreg_residuals.png",            "Residual Plot",                 "Error distribution around zero"),
        ("10_linreg_coefficients.png",         "LR Feature Coefficients",       "Feature importance for marks"),
        ("11_logreg_confusion_matrix.png",     "Confusion Matrix",              "Classification result breakdown"),
        ("12_logreg_probability.png",          "Pass Probability Distribution", "Separation between classes"),
        # also try src/models.py names
        ("8_linear_regression_actual_vs_pred.png", "Actual vs Predicted (LR)",  "Linear Regression performance"),
        ("9_linear_regression_residuals.png",      "Residual Plot",              "Error distribution around zero"),
        ("10_lr_feature_coefficients.png",         "LR Feature Coefficients",    "Feature importance for marks"),
        ("11_logistic_confusion_matrix.png",       "Confusion Matrix",           "Classification result breakdown"),
        ("12_logistic_probability_dist.png",       "Pass Probability Dist.",     "Separation between classes"),
        ("13_logistic_feature_coefficients.png",   "Logistic Coefficients",      "Feature log-odds importance"),
        # eda.py names
        ("1_feature_distributions.png",        "Feature Distributions",         "Histogram + KDE of each feature"),
        ("2_study_hours_vs_marks.png",         "Study Hours vs Final Marks",    "Shows the positive correlation"),
        ("3_attendance_vs_marks.png",          "Attendance vs Final Marks",     "Higher attendance → better marks"),
        ("4_correlation_heatmap.png",          "Correlation Heatmap",           "Strength of relationships"),
        ("5_boxplots_pass_fail.png",           "Box Plots: Pass vs Fail",       "Feature spread by outcome"),
        ("6_pair_plot.png",                    "Pair Plot",                     "All-feature pairwise scatter"),
        ("7_pass_fail_count.png",              "Pass / Fail Count",             "Class distribution"),
    ]

    seen_titles = set()
    items = []
    for fname, title, desc in plot_map:
        if title in seen_titles:
            continue
        path = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(path):
            seen_titles.add(title)
            b64  = img_to_b64(path)
            items.append(f"""
            <div class="plot-card">
                <img src="{b64}" alt="{title}" loading="lazy"/>
                <div class="plot-info">
                    <div class="plot-title">{title}</div>
                    <div class="plot-desc">{desc}</div>
                </div>
            </div>""")

    return "\n".join(items)


def generate():
    print("📊  Generating HTML report …")
    mae, rmse, r2, acc, prec, rec, f1, pass_rate = get_metrics()

    metrics_ok = mae is not None
    now = datetime.datetime.now().strftime("%d %B %Y, %H:%M")

    m_mae   = f"{mae:.2f}"   if metrics_ok else "N/A"
    m_rmse  = f"{rmse:.2f}"  if metrics_ok else "N/A"
    m_r2    = f"{r2:.4f}"    if metrics_ok else "N/A"
    m_acc   = f"{acc*100:.1f}%" if metrics_ok else "N/A"
    m_prec  = f"{prec*100:.1f}%" if metrics_ok else "N/A"
    m_rec   = f"{rec*100:.1f}%"  if metrics_ok else "N/A"
    m_f1    = f"{f1:.4f}"    if metrics_ok else "N/A"
    m_pass  = f"{pass_rate:.1f}%" if metrics_ok else "N/A"

    metric_cards = "".join([
        card("MAE",        m_mae,  "Mean Absolute Error (marks)", "#4C72B0"),
        card("RMSE",       m_rmse, "Root Mean Squared Error",     "#8172B2"),
        card("R² Score",   m_r2,   "Variance Explained",          "#55A868"),
        card("Accuracy",   m_acc,  "Pass/Fail Accuracy",          "#DD8452"),
        card("Precision",  m_prec, "Predicted Pass — True Pass",  "#4C72B0"),
        card("Recall",     m_rec,  "Actual Pass — Caught",        "#55A868"),
        card("F1-Score",   m_f1,   "Harmonic Mean P & R",         "#C44E52"),
        card("Pass Rate",  m_pass, "Overall student pass rate",   "#55A868"),
    ])

    gallery = build_plot_gallery()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Student Performance Prediction — Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
  <style>
    :root {{
      --bg:      #0d1117;
      --surface: #161b22;
      --card:    #1c2230;
      --border:  #30363d;
      --primary: #4C72B0;
      --accent:  #55A868;
      --red:     #C44E52;
      --orange:  #DD8452;
      --text:    #e6edf3;
      --muted:   #8b949e;
      --radius:  14px;
    }}
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
    }}

    /* ── HERO ── */
    .hero {{
      background: linear-gradient(135deg, #0d1117 0%, #161b40 50%, #0d1117 100%);
      border-bottom: 1px solid var(--border);
      padding: 60px 5% 50px;
      text-align: center;
      position: relative;
      overflow: hidden;
    }}
    .hero::before {{
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(76,114,176,.18) 0%, transparent 70%);
      pointer-events: none;
    }}
    .hero-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(76,114,176,.15);
      border: 1px solid rgba(76,114,176,.4);
      border-radius: 100px;
      padding: 6px 18px;
      font-size: .78rem;
      font-weight: 600;
      letter-spacing: .06em;
      text-transform: uppercase;
      color: #7eaaff;
      margin-bottom: 22px;
    }}
    .hero h1 {{
      font-size: clamp(1.8rem, 4vw, 3rem);
      font-weight: 800;
      letter-spacing: -.02em;
      margin-bottom: 14px;
      background: linear-gradient(135deg, #e6edf3 30%, #7eaaff 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    .hero p {{
      color: var(--muted);
      font-size: 1.05rem;
      max-width: 600px;
      margin: 0 auto 8px;
    }}
    .hero-meta {{
      font-size: .82rem;
      color: var(--muted);
      margin-top: 10px;
    }}

    /* ── LAYOUT ── */
    .container {{ max-width: 1280px; margin: 0 auto; padding: 0 5%; }}
    section {{ padding: 50px 0; }}
    .section-label {{
      font-size: .7rem;
      font-weight: 700;
      letter-spacing: .1em;
      text-transform: uppercase;
      color: var(--primary);
      margin-bottom: 6px;
    }}
    .section-title {{
      font-size: 1.6rem;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .section-sub {{
      color: var(--muted);
      font-size: .95rem;
      margin-bottom: 36px;
    }}
    hr.divider {{ border: none; border-top: 1px solid var(--border); margin: 0; }}

    /* ── METRIC CARDS ── */
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 18px;
    }}
    .metric-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 22px 20px;
      transition: transform .2s, box-shadow .2s;
    }}
    .metric-card:hover {{
      transform: translateY(-4px);
      box-shadow: 0 8px 30px rgba(0,0,0,.4);
    }}
    .metric-title {{
      font-size: .78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .metric-value {{
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: -.03em;
      margin-bottom: 6px;
    }}
    .metric-sub {{
      font-size: .75rem;
      color: var(--muted);
    }}

    /* ── PIPELINE ── */
    .pipeline {{
      display: flex;
      gap: 0;
      overflow-x: auto;
      padding-bottom: 10px;
    }}
    .pipe-step {{
      flex: 1;
      min-width: 140px;
      background: var(--card);
      border: 1px solid var(--border);
      border-right: none;
      padding: 20px 16px;
      position: relative;
      transition: background .2s;
    }}
    .pipe-step:first-child {{ border-radius: var(--radius) 0 0 var(--radius); }}
    .pipe-step:last-child  {{ border-right: 1px solid var(--border); border-radius: 0 var(--radius) var(--radius) 0; }}
    .pipe-step:hover {{ background: #222b3a; }}
    .pipe-num {{
      width: 30px; height: 30px;
      border-radius: 50%;
      background: var(--primary);
      color: #fff;
      font-size: .8rem;
      font-weight: 700;
      display: flex; align-items: center; justify-content: center;
      margin-bottom: 10px;
    }}
    .pipe-name {{
      font-size: .85rem;
      font-weight: 600;
      margin-bottom: 4px;
    }}
    .pipe-desc {{
      font-size: .73rem;
      color: var(--muted);
      line-height: 1.4;
    }}

    /* ── PLOT GALLERY ── */
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 22px;
    }}
    .plot-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      transition: transform .25s, box-shadow .25s;
    }}
    .plot-card:hover {{
      transform: translateY(-5px);
      box-shadow: 0 12px 40px rgba(0,0,0,.5);
    }}
    .plot-card img {{
      width: 100%;
      display: block;
      max-height: 280px;
      object-fit: cover;
      background: #fff;
    }}
    .plot-info {{
      padding: 14px 16px;
      border-top: 1px solid var(--border);
    }}
    .plot-title {{
      font-size: .9rem;
      font-weight: 600;
      margin-bottom: 4px;
    }}
    .plot-desc {{
      font-size: .77rem;
      color: var(--muted);
    }}

    /* ── MODEL EXPLAIN ── */
    .model-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 22px;
    }}
    .model-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 28px 26px;
    }}
    .model-icon {{ font-size: 2rem; margin-bottom: 12px; }}
    .model-name {{
      font-size: 1.1rem;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .model-desc {{
      font-size: .87rem;
      color: var(--muted);
      line-height: 1.6;
      margin-bottom: 16px;
    }}
    .tag {{
      display: inline-block;
      background: rgba(76,114,176,.18);
      border: 1px solid rgba(76,114,176,.35);
      color: #7eaaff;
      border-radius: 100px;
      padding: 3px 12px;
      font-size: .72rem;
      font-weight: 600;
      margin-right: 6px;
      margin-bottom: 6px;
    }}
    .tag.green {{
      background: rgba(85,168,104,.18);
      border-color: rgba(85,168,104,.35);
      color: #6ecf87;
    }}

    /* ── FEATURES TABLE ── */
    .tbl-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: .87rem;
    }}
    th {{
      background: var(--surface);
      padding: 12px 16px;
      text-align: left;
      font-size: .75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .05em;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
    }}
    td {{
      padding: 11px 16px;
      border-bottom: 1px solid var(--border);
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255,255,255,.03); }}
    .badge {{
      display: inline-block;
      padding: 2px 10px;
      border-radius: 100px;
      font-size: .72rem;
      font-weight: 600;
    }}
    .badge.num  {{ background:rgba(76,114,176,.2);  color:#7eaaff; }}
    .badge.cat  {{ background:rgba(221,132,82,.2);  color:#f0a87a; }}
    .badge.tgt  {{ background:rgba(85,168,104,.2);  color:#6ecf87; }}

    /* ── FOOTER ── */
    footer {{
      background: var(--surface);
      border-top: 1px solid var(--border);
      padding: 30px 5%;
      text-align: center;
      font-size: .82rem;
      color: var(--muted);
    }}
    footer strong {{ color: var(--text); }}

    /* ── ANIMATIONS ── */
    @keyframes fadeUp {{
      from {{ opacity:0; transform:translateY(24px); }}
      to   {{ opacity:1; transform:translateY(0);    }}
    }}
    .hero, .metrics-grid, .pipeline, .plot-grid, .model-grid {{
      animation: fadeUp .6s ease both;
    }}
  </style>
</head>
<body>

<!-- ═══════════ HERO ═══════════ -->
<div class="hero">
  <div class="hero-badge">🎓 Final Year Data Science Project (FDS)</div>
  <h1>Student Performance<br/>Prediction System</h1>
  <p>A complete ML pipeline — data generation, preprocessing, EDA,
     Linear &amp; Logistic Regression, evaluation, and interactive prediction.</p>
  <div class="hero-meta">Report generated: {now}</div>
</div>

<!-- ═══════════ PIPELINE ═══════════ -->
<div class="container">
<section>
  <div class="section-label">Pipeline Overview</div>
  <div class="section-title">End-to-End Workflow</div>
  <div class="section-sub">Seven sequential steps from raw data to interactive prediction.</div>
  <div class="pipeline">
    <div class="pipe-step">
      <div class="pipe-num">1</div>
      <div class="pipe-name">Generate Dataset</div>
      <div class="pipe-desc">500 synthetic students with 7 features &amp; realistic noise</div>
    </div>
    <div class="pipe-step">
      <div class="pipe-num">2</div>
      <div class="pipe-name">Preprocessing</div>
      <div class="pipe-desc">Fill NaN, label-encode, Min-Max normalise</div>
    </div>
    <div class="pipe-step">
      <div class="pipe-num">3</div>
      <div class="pipe-name">EDA</div>
      <div class="pipe-desc">7 charts revealing patterns and correlations</div>
    </div>
    <div class="pipe-step">
      <div class="pipe-num">4</div>
      <div class="pipe-name">Train Models</div>
      <div class="pipe-desc">Linear Regression + Logistic Regression</div>
    </div>
    <div class="pipe-step">
      <div class="pipe-num">5</div>
      <div class="pipe-name">Evaluate</div>
      <div class="pipe-desc">MAE, R², Accuracy, F1, Confusion Matrix</div>
    </div>
    <div class="pipe-step">
      <div class="pipe-num">6</div>
      <div class="pipe-name">Save Models</div>
      <div class="pipe-desc">Pickle .pkl files for reuse</div>
    </div>
    <div class="pipe-step">
      <div class="pipe-num">7</div>
      <div class="pipe-name">Predict</div>
      <div class="pipe-desc">Interactive console input &amp; personalised advice</div>
    </div>
  </div>
</section>
<hr class="divider"/>

<!-- ═══════════ METRICS ═══════════ -->
<section>
  <div class="section-label">Model Performance</div>
  <div class="section-title">Evaluation Metrics</div>
  <div class="section-sub">Results measured on the 20% held-out test set (100 samples).</div>
  <div class="metrics-grid">
    {metric_cards}
  </div>
</section>
<hr class="divider"/>

<!-- ═══════════ FEATURES TABLE ═══════════ -->
<section>
  <div class="section-label">Dataset</div>
  <div class="section-title">Features &amp; Targets</div>
  <div class="section-sub">500 students · 7 input features · 2 target variables</div>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>Column</th>
          <th>Type</th>
          <th>Range</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>study_hours</td>     <td><span class="badge num">Numeric</span></td><td>0.5 – 10 hrs</td>   <td>Daily study hours</td></tr>
        <tr><td>attendance_pct</td>  <td><span class="badge num">Numeric</span></td><td>50 – 100 %</td>    <td>Class attendance percentage</td></tr>
        <tr><td>prev_marks</td>      <td><span class="badge num">Numeric</span></td><td>30 – 100</td>      <td>Previous semester marks</td></tr>
        <tr><td>sleep_hours</td>     <td><span class="badge num">Numeric</span></td><td>4 – 10 hrs</td>    <td>Average sleep per night</td></tr>
        <tr><td>gender</td>          <td><span class="badge cat">Categorical</span></td><td>Male / Female</td><td>Student gender (encoded 0/1)</td></tr>
        <tr><td>extracurricular</td> <td><span class="badge cat">Categorical</span></td><td>Yes / No</td>   <td>Participates in extra activities</td></tr>
        <tr><td>internet_access</td> <td><span class="badge cat">Categorical</span></td><td>Yes / No</td>   <td>Internet at home (encoded 1/0)</td></tr>
        <tr><td>final_marks</td>     <td><span class="badge tgt">Target</span></td><td>0 – 100</td>        <td>🎯 Regression target (predicted score)</td></tr>
        <tr><td>pass_fail</td>       <td><span class="badge tgt">Target</span></td><td>0 / 1</td>          <td>🎯 Classification target (≥40 = Pass)</td></tr>
      </tbody>
    </table>
  </div>
</section>
<hr class="divider"/>

<!-- ═══════════ MODEL EXPLANATION ═══════════ -->
<section>
  <div class="section-label">Machine Learning</div>
  <div class="section-title">Models Used</div>
  <div class="section-sub">Two complementary algorithms for regression and classification tasks.</div>
  <div class="model-grid">
    <div class="model-card">
      <div class="model-icon">📈</div>
      <div class="model-name">Linear Regression</div>
      <div class="model-desc">
        Fits the best straight line through the data by minimising the
        sum of squared errors. Predicts a <strong>continuous</strong> final
        marks score (0–100). We evaluate it with MAE, RMSE, and R².
      </div>
      <span class="tag">sklearn.linear_model</span>
      <span class="tag">Regression</span>
      <span class="tag green">Final Marks</span>
    </div>
    <div class="model-card">
      <div class="model-icon">🎯</div>
      <div class="model-name">Logistic Regression</div>
      <div class="model-desc">
        Uses a sigmoid function to model the probability of a student
        passing. Classifies each student as <strong>Pass (1)</strong> or
        <strong>Fail (0)</strong> based on a 0.5 decision boundary.
      </div>
      <span class="tag">sklearn.linear_model</span>
      <span class="tag">Classification</span>
      <span class="tag green">Pass / Fail</span>
    </div>
    <div class="model-card">
      <div class="model-icon">🔧</div>
      <div class="model-name">Preprocessing Pipeline</div>
      <div class="model-desc">
        Raw data is cleaned by filling missing numeric values with the
        column <strong>median</strong>, encoding categorical columns via
        label encoding, then scaling all features to [0, 1] with
        <strong>Min-Max normalisation</strong>.
      </div>
      <span class="tag">pandas</span>
      <span class="tag">MinMaxScaler</span>
      <span class="tag green">80/20 Split</span>
    </div>
  </div>
</section>
<hr class="divider"/>

<!-- ═══════════ EDA GALLERY ═══════════ -->
<section>
  <div class="section-label">Visualisations</div>
  <div class="section-title">Charts &amp; Plots</div>
  <div class="section-sub">All plots generated during EDA and model evaluation stages.</div>
  <div class="plot-grid">
    {gallery}
  </div>
</section>
<hr class="divider"/>

<!-- ═══════════ KEY INSIGHTS ═══════════ -->
<section>
  <div class="section-label">Findings</div>
  <div class="section-title">Key Insights</div>
  <div class="section-sub">Observations derived from exploratory data analysis.</div>
  <div class="model-grid" style="grid-template-columns: repeat(auto-fill, minmax(240px,1fr));">
    <div class="model-card">
      <div class="model-icon">📚</div>
      <div class="model-name">Study Hours Matter Most</div>
      <div class="model-desc">The strongest positive predictor of final marks. Each extra hour of daily study shows a significant mark improvement.</div>
    </div>
    <div class="model-card">
      <div class="model-icon">📅</div>
      <div class="model-name">Attendance is Critical</div>
      <div class="model-desc">Students with lower attendance consistently score below the pass threshold regardless of other factors.</div>
    </div>
    <div class="model-card">
      <div class="model-icon">🔁</div>
      <div class="model-name">Previous Marks Predict Future</div>
      <div class="model-desc">Past academic performance has the highest single-feature correlation with final marks, showing momentum matters.</div>
    </div>
    <div class="model-card">
      <div class="model-icon">😴</div>
      <div class="model-name">Sleep Helps Marginally</div>
      <div class="model-desc">Sleep hours show a small but positive influence. Students sleeping 7–8 hrs perform slightly better than extremes.</div>
    </div>
    <div class="model-card">
      <div class="model-icon">🌐</div>
      <div class="model-name">Internet Access Positive</div>
      <div class="model-desc">Having internet at home correlates with higher mark predictions due to access to resources and study materials.</div>
    </div>
    <div class="model-card">
      <div class="model-icon">🏆</div>
      <div class="model-name">Model Accuracy High</div>
      <div class="model-desc">Logistic Regression achieves strong accuracy on the test set, validating the dataset's predictive signal.</div>
    </div>
  </div>
</section>
</div>

<!-- ═══════════ FOOTER ═══════════ -->
<footer>
  <strong>Student Performance Prediction System</strong> · Final Year Data Science Project (FDS)<br/>
  Built with Python · pandas · scikit-learn · matplotlib · seaborn<br/>
  Report generated on {now}
</footer>

</body>
</html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅  Report saved → {OUT_HTML}")
    print(f"   Open in any browser to view the interactive report.")


if __name__ == "__main__":
    generate()
