# ml_service.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# ✅ โหลดโมเดลและ label encoder
model = joblib.load("rf_multi_model.pkl")
le_stress = joblib.load("rf_multi_label_encoder_stress.pkl")
le_activity = joblib.load("rf_multi_label_encoder_activity.pkl")

# ✅ สร้าง FastAPI app
app = FastAPI(
    title="Stress Activity Prediction API",
    description="API สำหรับทำนายระดับความเครียดและกิจกรรมแนะนำ",
    version="1.0.0"
)

# ✅ กำหนดรูปแบบข้อมูล input
class InputData(BaseModel):
    total_score: int

# ✅ ทดสอบว่า API ทำงาน
@app.get("/")
def read_root():
    return {"message": "ML Service is running!"}

# ✅ Endpoint สำหรับทำนาย
@app.post("/predict")
def predict(data: InputData):
    try:
        # ✅ สร้าง DataFrame ที่มีชื่อคอลัมน์ตรงกับตอนที่ fit โมเดล
        input_df = pd.DataFrame([[data.total_score]], columns=["Total score"])

        # ✅ ทำนายผล
        prediction = model.predict(input_df)[0]

        # ✅ prediction ที่ได้ออกมาเป็น [stress, activity]
        stress_label = le_stress.inverse_transform([prediction[0]])[0]
        activity_label = le_activity.inverse_transform([prediction[1]])[0]

        return {
            "stress_level": stress_label,
            "recommended_activity": activity_label
        }
    except Exception as e:
        return {"error": str(e)}

# ✅ ถ้ารันไฟล์นี้ตรง ๆ
if __name__ == "__main__":
    # รันที่ port 8000
    uvicorn.run("ml_service:app", host="0.0.0.0", port=8000, reload=True)
