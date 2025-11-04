from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Load your trained LSTM model
model = joblib.load("lstm_model.pkl")

# Define input data schema
class InputData(BaseModel):
    features: list

# Define root endpoint
@app.get("/")
def home():
    return {"message": "LSTM Price/Demand Forecasting API is running!"}

# Define prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array([data.features])
        prediction = model.predict(features)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
