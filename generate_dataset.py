"""
generate_dataset.py
--------------------
Generates a realistic synthetic student performance dataset
and saves it as 'student_data.csv' in the same folder.

Columns:
  - student_id       : Unique identifier
  - gender           : Male / Female
  - study_hours      : Hours studied per day (0.5 – 10)
  - attendance_pct   : Attendance percentage (50 – 100)
  - prev_marks       : Previous semester marks (30 – 100)
  - sleep_hours      : Average sleep hours (4 – 10)
  - extracurricular  : Participates in extracurricular? (Yes/No)
  - internet_access  : Has internet at home? (Yes/No)
  - final_marks      : Final exam marks (target for regression)
  - pass_fail        : Pass (1) if final_marks >= 40 else Fail (0)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 500  # Number of students

student_id     = [f"S{str(i).zfill(4)}" for i in range(1, N + 1)]
gender         = np.random.choice(["Male", "Female"], size=N, p=[0.52, 0.48])
study_hours    = np.round(np.random.uniform(0.5, 10, N), 1)
attendance_pct = np.round(np.random.uniform(50, 100, N), 1)
prev_marks     = np.round(np.random.uniform(30, 100, N), 1)
sleep_hours    = np.round(np.random.uniform(4, 10, N), 1)
extracurricular= np.random.choice(["Yes", "No"], size=N, p=[0.4, 0.6])
internet_access= np.random.choice(["Yes", "No"], size=N, p=[0.7, 0.3])

# Final marks is influenced by study hours, attendance, and previous marks
noise = np.random.normal(0, 5, N)
final_marks = (
    0.35 * study_hours * 5    +   # study hours contribute heavily
    0.25 * attendance_pct * 0.6 + # attendance moderately
    0.30 * prev_marks * 0.7   +   # previous performance
    0.05 * sleep_hours * 2    +   # sleep helps a bit
    noise
)
final_marks = np.clip(np.round(final_marks, 1), 0, 100)
pass_fail = (final_marks >= 40).astype(int)

# Introduce ~3% missing values for realism
def inject_nulls(arr, pct=0.03):
    arr = arr.astype(object)
    idx = np.random.choice(len(arr), size=int(len(arr) * pct), replace=False)
    arr[idx] = np.nan
    return arr

df = pd.DataFrame({
    "student_id"       : student_id,
    "gender"           : gender,
    "study_hours"      : inject_nulls(study_hours),
    "attendance_pct"   : inject_nulls(attendance_pct),
    "prev_marks"       : inject_nulls(prev_marks),
    "sleep_hours"      : inject_nulls(sleep_hours),
    "extracurricular"  : extracurricular,
    "internet_access"  : internet_access,
    "final_marks"      : final_marks,
    "pass_fail"        : pass_fail,
})

out_path = os.path.join(os.path.dirname(__file__), "student_data.csv")
df.to_csv(out_path, index=False)
print(f"✅ Dataset saved → {out_path}  ({len(df)} rows)")
print(df.head())
