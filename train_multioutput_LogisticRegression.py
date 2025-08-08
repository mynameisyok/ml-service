# train_multioutput_LogisticRegression.py
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------- 1) โหลดข้อมูล ----------
df = pd.read_csv("DASS_Stress_Recommended_ByScore.csv")

# ---------- 2) เข้ารหัส Label ----------
le_stress = LabelEncoder()
le_activity = LabelEncoder()

df["Stress_Label"] = le_stress.fit_transform(df["Stress_Level"])
df["Activity_Label"] = le_activity.fit_transform(df["Recommended_Activity"])

# ---------- 3) เตรียม X และ y ----------
X = df[["Total score"]]
y = df[["Stress_Label", "Activity_Label"]]

# ---------- 4) train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y["Stress_Label"]
)

# ---------- 5) สร้างโมเดล LogisticRegression แบบ Multi-Output ----------
base_model = LogisticRegression(max_iter=100000)
multi_model = MultiOutputClassifier(base_model)
multi_model.fit(X_train, y_train)

# ---------- 6) ประเมินโมเดล ----------
y_pred = multi_model.predict(X_test)
acc_stress = accuracy_score(y_test["Stress_Label"], y_pred[:, 0])
acc_activity = accuracy_score(y_test["Activity_Label"], y_pred[:, 1])

print(f"📊 Accuracy  (Stress_Level)        : {acc_stress:.4f}")
print(f"📊 Accuracy  (Recommended Activity): {acc_activity:.4f}")

# ---------- 6.1) รายงานแบบละเอียด ----------
print("\n📋 Classification Report - Stress Level")
print(classification_report(
    y_test["Stress_Label"],
    y_pred[:, 0],
    target_names=le_stress.classes_,
    zero_division=0
))

print("\n📋 Classification Report - Recommended Activity")
actual_labels = np.unique(y_test["Activity_Label"])
print(classification_report(
    y_test["Activity_Label"],
    y_pred[:, 1],
    labels=actual_labels,
    target_names=le_activity.inverse_transform(actual_labels),
    zero_division=0
))

# ---------- 7) ฟังก์ชันทำนาย ----------
def predict(total_score: int):
    pred_enc = multi_model.predict(np.array([[total_score]]))[0]
    stress_label = le_stress.inverse_transform([pred_enc[0]])[0]
    act_label = le_activity.inverse_transform([pred_enc[1]])[0]
    return stress_label, act_label

# ---------- 8) บันทึกโมเดล ----------
joblib.dump(multi_model, "lr_multi_model.pkl")
joblib.dump(le_stress, "lr_multi_label_encoder_stress.pkl")
joblib.dump(le_activity, "lr_multi_label_encoder_activity.pkl")

print("✅ Logistic Regression model trained and saved.")
