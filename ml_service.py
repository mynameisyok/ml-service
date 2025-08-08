# ml_service_knn.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# ✅ โหลดโมเดล KNN และ label encoders
model = joblib.load("knn_multi_model.pkl")
le_stress = joblib.load("knn_multi_label_encoder_stress.pkl")
le_activity = joblib.load("knn_multi_label_encoder_activity.pkl")

# ✅ สร้าง FastAPI app
app = FastAPI(
    title="Stress Activity Prediction API (KNN)",
    description="API สำหรับทำนายระดับความเครียดและกิจกรรมแนะนำ ด้วยโมเดล KNN",
    version="1.0.0"
)

# ✅ กำหนดรูปแบบข้อมูล input
class InputData(BaseModel):
    total_score: int

# ✅ ทดสอบว่า API ทำงาน
@app.get("/")
def read_root():
    return {"message": "KNN ML Service is running!"}

# ✅ Endpoint สำหรับทำนาย
@app.post("/predict")
def predict(data: InputData):
    try:
        # ต้องใช้ชื่อคอลัมน์ให้ตรงกับตอน train ("Total score")
        input_df = pd.DataFrame([[data.total_score]], columns=["Total score"])

        # ทำนาย: คืนค่าเป็น [stress_label_encoded, activity_label_encoded]
        pred = model.predict(input_df)[0]

        stress_label = le_stress.inverse_transform([pred[0]])[0]
        activity_label = le_activity.inverse_transform([pred[1]])[0]

        return {
            "stress_level": stress_label,
            "recommended_activity": activity_label
        }
    except Exception as e:
        return {"error": str(e)}

# ✅ รันตรง ๆ
if __name__ == "__main__":
    # รันที่พอร์ต 8001 แยกจาก RF (ปรับได้ตามต้องการ)
    uvicorn.run("ml_service_knn:app", host="0.0.0.0", port=8001, reload=True)
