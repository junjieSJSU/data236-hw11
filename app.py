from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

# Load model
model = joblib.load("iris_classifier.pkl")

# Iris class names (consistent with sklearn's Iris dataset)
class_names = ["setosa", "versicolor", "virginica"]

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

@app.post("/predict")
def predict_iris(data: IrisInput):
    try:
        features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
        pred_index = model.predict(features)[0]
        prediction = class_names[pred_index]

        return {
            "input": data.dict(),
            "predicted_class": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
