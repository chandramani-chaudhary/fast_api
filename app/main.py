#Api code
import pandas as pd
import os
from app.model import load_model_scaler
from app.schema import EmployeeStatus
from fastapi import FastAPI

app = FastAPI()

print("Current working dir:", os.getcwd())
print("Files:", os.listdir())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(BASE_DIR, '..', 'data', 'HR_Dataset_Refresh.csv')

df = pd.read_csv(file_path)
model, scaler = load_model_scaler()

@app.get("/")
def home():
    return "Welcome to fast Api employee statues project!"
@app.post("/predict-status")
def predict_status(data:EmployeeStatus):
    input_data = pd.DataFrame([
        data.model_dump()
    ])

    input_scaler = scaler.transform(input_data)
    prediction = model.predict(input_scaler)[0]
    return{
        "Predicted_status":int(prediction),
        "Status":"likely to be Terminated !!" if prediction==1 else "likely to be Activate!!"
    }
