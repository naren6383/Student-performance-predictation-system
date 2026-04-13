"""
predict.py
-----------
Step 7 — Interactive User Prediction

What this does:
  ✅ Loads the trained models from disk
  ✅ Asks the user for their details (study hours, attendance, etc.)
  ✅ Applies the same normalisation used during training
  ✅ Predicts:
       • Estimated Final Marks  (Linear Regression)
       • Pass or Fail           (Logistic Regression)
  ✅ Prints recommendations based on the prediction

Simple explanation:
  Once training is done, we can use the model like a calculator.
  You provide inputs → the model applies the equation → you get a prediction.
  We normalise your input the same way we normalised training data.
"""

import os
import pickle
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE, "outputs", "models")
DATA_PATH = os.path.join(BASE, "data",    "student_cleaned.csv")


def load_model(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Please run main.py first to train the models."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def get_normalisation_params(data_path: str = DATA_PATH):
    """
    Compute min/max from the cleaned training data so we can
    apply the SAME normalisation to user inputs.
    """
    df = pd.read_csv(data_path)
    cols = ["study_hours", "attendance_pct", "prev_marks",
            "sleep_hours", "gender", "extracurricular", "internet_access"]
    params = {}
    for c in cols:
        params[c] = (df[c].min(), df[c].max())
    return params


def normalise(value, col_min, col_max):
    """Min-Max normalisation for a single value."""
    if col_max == col_min:
        return 0.0
    return (value - col_min) / (col_max - col_min)


def get_float_input(prompt: str, lo: float, hi: float) -> float:
    """Prompt until user enters a valid float in [lo, hi]."""
    while True:
        try:
            val = float(input(prompt))
            if lo <= val <= hi:
                return val
            print(f"  ⚠  Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("  ⚠  Invalid input. Please enter a number.")


def get_choice_input(prompt: str, choices: list) -> str:
    """Prompt until user picks one of the valid choices."""
    while True:
        val = input(prompt).strip().title()
        if val in choices:
            return val
        print(f"  ⚠  Choose from: {choices}")


def build_feature_vector(params: dict) -> np.ndarray:
    """Collect inputs from the user and return a normalised feature array."""
    print("\n" + "═" * 55)
    print("  📝  Student Details Input")
    print("═" * 55)

    study_h = get_float_input(
        "  Daily study hours      (0.5 – 10.0): ", 0.5, 10.0)
    attend  = get_float_input(
        "  Attendance percentage  (50 – 100)  : ", 50.0, 100.0)
    prev_m  = get_float_input(
        "  Previous semester marks(30 – 100)  : ", 30.0, 100.0)
    sleep_h = get_float_input(
        "  Sleep hours per night  (4 – 10)    : ", 4.0, 10.0)
    gender  = get_choice_input(
        "  Gender                 (Male/Female): ", ["Male", "Female"])
    extra   = get_choice_input(
        "  Extracurricular        (Yes/No)    : ", ["Yes", "No"])
    internet= get_choice_input(
        "  Internet at home       (Yes/No)    : ", ["Yes", "No"])

    # Encode categoricals
    gender_enc = 1 if gender == "Female" else 0
    extra_enc  = 1 if extra  == "Yes"    else 0
    net_enc    = 1 if internet == "Yes"  else 0

    raw = {
        "study_hours"   : study_h,
        "attendance_pct": attend,
        "prev_marks"    : prev_m,
        "sleep_hours"   : sleep_h,
        "gender"        : float(gender_enc),
        "extracurricular": float(extra_enc),
        "internet_access": float(net_enc),
    }

    # Normalise using training data statistics
    norm_vals = []
    for col, val in raw.items():
        lo, hi = params[col]
        norm_vals.append(normalise(val, lo, hi))

    return np.array(norm_vals).reshape(1, -1), raw


def print_recommendations(pred_marks: float, pass_prob: float):
    """Print personalised recommendations based on prediction."""
    print("\n  📚  Personalised Recommendations:")

    if pred_marks < 40:
        tips = [
            "⚠  You are at risk of failing. Increase daily study hours.",
            "📅  Attend every class — attendance strongly affects marks.",
            "😴  Ensure 7-8 hours of sleep for better retention.",
            "🧑‍🏫  Consider joining a study group or seeking teacher support.",
        ]
    elif pred_marks < 60:
        tips = [
            "📈  You are likely to pass but there is room to improve.",
            "⏱  Add 1-2 extra study hours each day.",
            "🗒  Revise previous semester topics to strengthen your base.",
            "📶  Use internet resources (YouTube, Khan Academy) for concepts.",
        ]
    elif pred_marks < 80:
        tips = [
            "👍  Good performance expected! Keep the momentum going.",
            "🎯  Focus on weak topics for even better results.",
            "📝  Practice past exam papers to boost confidence.",
        ]
    else:
        tips = [
            "🏆  Excellent prediction! You are on track for distinction.",
            "🚀  Consider exploring advanced topics or helping peers.",
            "🎓  Maintain your study habits — consistency is key.",
        ]

    for t in tips:
        print(f"    {t}")


def predict():
    print("\n" + "═" * 55)
    print("  🎓  Student Performance Prediction System")
    print("═" * 55)

    # Load trained models
    try:
        lr_model = load_model("linear_regression.pkl")
        lc_model = load_model("logistic_regression.pkl")
        print("\n  ✅  Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"\n  ❌  {e}")
        return

    # Get normalisation ranges from cleaned data
    norm_params = get_normalisation_params()

    while True:
        feature_vec, raw_inputs = build_feature_vector(norm_params)

        # ── Predictions ──────────────────────────────────────────────────
        pred_marks   = lr_model.predict(feature_vec)[0]
        pred_marks   = float(np.clip(pred_marks, 0, 100))
        pass_proba   = lc_model.predict_proba(feature_vec)[0][1]
        pred_outcome = "✅ PASS" if pass_proba >= 0.5 else "❌ FAIL"

        # Display results
        print("\n" + "═" * 55)
        print("  🎯  PREDICTION RESULTS")
        print("═" * 55)
        print(f"\n  Predicted Final Marks : {pred_marks:.1f} / 100")
        print(f"  Pass Probability      : {pass_proba*100:.1f}%")
        print(f"  Expected Outcome      : {pred_outcome}\n")

        bar_len = int(pass_proba * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  Pass Confidence [{bar}] {pass_proba*100:.0f}%")

        print_recommendations(pred_marks, pass_proba)

        # ── Ask to continue ───────────────────────────────────────────────
        print("\n" + "─" * 55)
        again = input("  Predict for another student? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\n  🎓  Thank you for using Student Performance Prediction System!\n")
            break


if __name__ == "__main__":
    predict()
