from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# Create FastAPI instance
app = FastAPI(title="CommerceSense FastAPI Gateway")

# Define input schema (same as model expects)
class InputData(BaseModel):
    Selling_Price: float
    Discount_Percentage: float
    Stock_Availability: float
    Purchase_Frequency: float
    Quantity_Sold_Lagged: float
    Season_Summer: int
    Season_Winter: int

# 🔹 Replace this URL with your deployed Flask API endpoint
FLASK_MODEL_URL = "https://your-flask-model.onrender.com/predict"

@app.get("/")
def home():
    return {"message": "FastAPI Gateway is running 🚀"}

@app.post("/predict")
def get_prediction(data: InputData):
    try:
        # Convert FastAPI input into Flask API format
        flask_payload = {
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

        # Send POST request to Flask model API
        response = requests.post(FLASK_MODEL_URL, json=flask_payload)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Flask API error: {response.text}")

        prediction = response.json().get("prediction", [])
        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

