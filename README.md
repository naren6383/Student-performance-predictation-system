# 🎓 Student Performance Prediction System
### Final Year Data Science Project (FDS)

A complete, end-to-end machine learning pipeline that predicts student academic performance using **Linear Regression** (marks) and **Logistic Regression** (pass/fail).

---

## 📁 Project Structure

```
student perfomene predtcn sytm (FDS)/
│
├── main.py                  ← ✅ Run this to execute the full pipeline
├── generate_report.py       ← ✅ Run after main.py to get an HTML report
├── requirements.txt         ← Python dependencies
├── README.md                ← This file
│
├── data/
│   ├── student_data.csv     ← Raw generated dataset (500 students)
│   └── student_cleaned.csv  ← Preprocessed dataset
│
├── src/
│   ├── preprocessing.py     ← Step 2: Data cleaning & normalisation
│   ├── eda.py               ← Step 3: Exploratory Data Analysis
│   ├── models.py            ← Steps 4–6: Train & evaluate models
│   └── predict.py           ← Step 7: Interactive prediction (standalone)
│
└── outputs/
    ├── plots/               ← All charts (PNG) saved here
    ├── models/              ← Trained model pickles (.pkl)
    └── report.html          ← ⭐ Beautiful HTML report (after generate_report.py)
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python main.py
```

This will:
- Generate 500 synthetic student records
- Preprocess the data (fill nulls, encode, normalise)
- Run 7 EDA visualisations and save them to `outputs/plots/`
- Train and evaluate Linear & Logistic Regression models
- Launch an **interactive prediction** interface

### 3. Generate the HTML Report
```bash
python generate_report.py
```
Opens `outputs/report.html` — a beautiful dark-themed summary with all charts embedded.

---

## 🔀 Run Individual Modules

```bash
# Only preprocessing
python src/preprocessing.py

# Only EDA (needs cleaned CSV)
python src/eda.py

# Only model training + evaluation
python src/models.py

# Only interactive prediction (needs trained models)
python src/predict.py
```

---

## 📊 Dataset

| Feature | Type | Range | Description |
|---|---|---|---|
| `study_hours` | Numeric | 0.5 – 10 hrs/day | Daily study hours |
| `attendance_pct` | Numeric | 50 – 100 % | Class attendance |
| `prev_marks` | Numeric | 30 – 100 | Previous semester marks |
| `sleep_hours` | Numeric | 4 – 10 hrs | Average sleep per night |
| `gender` | Categorical | Male / Female | Encoded 0 / 1 |
| `extracurricular` | Categorical | Yes / No | Extra activities |
| `internet_access` | Categorical | Yes / No | Internet at home |
| **`final_marks`** | **Target** | 0 – 100 | 🎯 Regression prediction |
| **`pass_fail`** | **Target** | 0 / 1 | 🎯 Classification (≥40 = Pass) |

- **500 students** generated synthetically with `numpy.random` + realistic noise
- ~3% missing values injected to simulate real-world data

---

## 🤖 Models

### Linear Regression — Predict Final Marks
- Fits a weighted linear equation to the 7 features
- Minimises Mean Squared Error (MSE)
- Metrics: **MAE**, **RMSE**, **R² Score**

### Logistic Regression — Predict Pass / Fail
- Applies sigmoid function to model P(Pass)
- Decision boundary at probability = 0.5
- Metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, **Confusion Matrix**

---

## 📈 EDA Charts Generated

| # | Plot | Description |
|---|---|---|
| 1 | Feature Distributions | Histogram + KDE per feature |
| 2 | Study Hours vs Marks | Scatter + trend line coloured by outcome |
| 3 | Attendance vs Marks | Same as above for attendance |
| 4 | Correlation Heatmap | All feature correlations at a glance |
| 5 | Box Plots | Feature spread by Pass / Fail |
| 6 | Pair Plot | Pairwise all-feature scatter matrix |
| 7 | Pass/Fail Count | Class distribution bar chart |
| 8 | Actual vs Predicted | Linear Regression scatter |
| 9 | Residual Plot | Errors should scatter around 0 |
| 10 | LR Coefficients | Feature impact on mark prediction |
| 11 | Confusion Matrix | TP / FP / TN / FN breakdown |
| 12 | Probability Distribution | Pass probability histogram by true class |
| 13 | Logistic Coefficients | Log-odds feature importance |

---

## 💡 Key Insights

- **Previous marks** have the highest single correlation with final marks  
- **Study hours** show the strongest positive impact on predicted scores  
- **Attendance** is critical — missing classes strongly predicts failure  
- **Sleep hours** have a small but positive influence  
- **Internet access** correlates with slightly better performance

---

## 🛠 Tech Stack

| Library | Version | Used For |
|---|---|---|
| `pandas` | ≥ 1.5 | Data loading, manipulation |
| `numpy` | ≥ 1.23 | Numerical operations |
| `matplotlib` | ≥ 3.6 | Plotting |
| `seaborn` | ≥ 0.12 | Statistical plots |
| `scikit-learn` | ≥ 1.1 | ML models, metrics, preprocessing |

Install all with:
```bash
pip install -r requirements.txt
```

---

## ⚙ Command-Line Flags (main.py)

| Flag | Effect |
|---|---|
| `python main.py` | Full pipeline (default) |
| `python main.py --skip-data` | Skip re-generating the dataset |
| `python main.py --skip-eda` | Skip EDA charts |
| `python main.py --no-predict` | Skip interactive prediction |

---

*Built with ❤️ for the Final Year Data Science (FDS) project.*
