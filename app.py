from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# creating fast app
app = FastAPI()
model = joblib.load("iris-classifier-week-1_model.joblib")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Iris Predictor API running!"}

@app.post("/predict")
def predict(data: IrisInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"predicted_class": prediction}

