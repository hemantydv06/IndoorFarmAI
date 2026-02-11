from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import uvicorn

app = FastAPI(title="IndoorFarmAI API v2.0")

# Load model function (same as Streamlit)
def load_model():
    np.random.seed(42)
    crops = ['lettuce', 'spinach', 'tomato', 'basil', 'kale']
    data = []
    # ... (same model loading code as above)
    return model, scaler

model, scaler = load_model()

class FarmRequest(BaseModel):
    space_m2: float
    temperature: float
    humidity: float
    budget: float
    location: str = "Airoli"

@app.post("/predict")
async def predict_crop(request: FarmRequest):
    conditions = [[30,25,40, request.temperature, request.humidity, 6.2, 90]]
    crop = model.predict(scaler.transform(conditions))[0]
    confidence = float(max(model.predict_proba(scaler.transform(conditions))[0]))
    
    prices = {'lettuce':45, 'spinach':38, 'tomato':65, 'basil':120, 'kale':42}
    profit = prices.get(crop.lower(), 50) * (2.0 if request.space_m2 < 10 else 1.5) * request.space_m2
    
    return {
        "recommended_crop": crop,
        "confidence": confidence,
        "estimated_profit": round(profit, 2),
        "farming_method": "hydroponics" if request.space_m2 < 10 else "soil",
        "space_efficiency": "high" if request.space_m2 < 10 else "medium"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
