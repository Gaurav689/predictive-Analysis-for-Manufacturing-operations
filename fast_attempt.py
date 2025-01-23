from fastapi import FastAPI, UploadFile, File
import pandas as pd
import pickle
import shutil
import os
import sys

# Add the current directory to the Python path
"""sys.path.append(os.getcwd()) 
from Model import train_model """

app = FastAPI()

# Define directory for uploaded files and saved model
UPLOAD_DIR = "uploads/" 
MODEL_PATH = "model.pkl"

@app.get("/")
async def root():
    return {"message": "Welcome to the API!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Create upload directory if it doesn't exist
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    file_path = os.path.join(UPLOAD_DIR, file.filename) 
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "File uploaded successfully"}

@app.post("/train")
async def train():
    from Model import train_model 
    filename = os.listdir(UPLOAD_DIR)[0]
    print (filename)
    file_path = os.path.join(UPLOAD_DIR, filename) 
    try:
        model = train_model(file_path) 
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f) 
        return {"message": "Model trained successfully"} 
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
async def predict(data: dict):
   try: 
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.DataFrame([data])
    print (df)
    prediction = model.predict(df)
    print(prediction)
    return {"Downtime": "Yes" if prediction[0] == 1 else "No"}
   except Exception as e:
    return {"error": str(e)}


