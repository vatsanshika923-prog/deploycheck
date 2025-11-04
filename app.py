from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(title="CommerceSense FastAPI Gateway")

# Replace with your actual Flask API URL
FLASK_MODEL_URL = "https://your-flask-model.onrender.com/predict"

class InputData(BaseModel):
    Selling_Price: float
    Discount_Percentage: float
    Stock_Availability: float
    Purchase_Frequency: float
    Quantity_Sold_Lagged: float
    Season_Summer: int
    Season_Winter: int

@app.get("/")
def home():
    return {"message": "FastAPI Gateway is running 🚀"}

@app.post("/predict")
def get_prediction(data: InputData):
    try:
        payload = {
            "input": [
                data.Selling_Price,
                data.Discount_Percentage,
                data.Stock_Availability,
                data.Purchase_Frequency,
                data.Quantity_Sold_Lagged,
                data.Season_Summer,
                data.Season_Winter
            ]
        }

        response = requests.post(FLASK_MODEL_URL, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Flask API error: {response.text}")

        result = response.json()
        return {"prediction": result.get("prediction", [])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

