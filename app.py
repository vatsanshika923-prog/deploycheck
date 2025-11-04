from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import traceback

# Initialize FastAPI
app = FastAPI(title="LSTM Price/Demand Forecasting API 🚀")

# ✅ Load trained model
try:
    model = joblib.load("lstm_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


# 🧾 Define input data structure
class ModelInput(BaseModel):
    Selling_Price: float
    Discount_Percentage: float
    Stock_Availability: float
    Purchase_Frequency: float
    Quantity_Sold_Lagged: float
    Season_Summer: int
    Season_Winter: int


# 🏠 Root route
@app.get("/")
def home():
    return {"message": "LSTM Forecasting API is running successfully 🚀"}


# 🔮 Prediction route
@app.post("/predict")
def predict(data: ModelInput):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Convert input into array format
        input_data = np.array([[
            data.Selling_Price,
            data.Discount_Percentage,
            data.Stock_Availability,
            data.Purchase_Frequency,
            data.Quantity_Sold_Lagged,
            data.Season_Summer,
            data.Season_Winter
        ]])

        # Make prediction
        prediction = model.predict(input_data)

        return {"prediction": float(prediction[0])}

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
