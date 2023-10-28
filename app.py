import uvicorn  # For Asynchronous server gateway interface which is multi-thread. Where as Fask uses WSGI which is Single-thread.
from fastapi import FastAPI  # Similar to Flask
from BankNotes import BankNote
import numpy as np
import pandas as pd
import pickle


# Create FastApi app object
app = FastAPI()

pickle_in = open("classfier.pkl", "rb")
classifier = pickle.load(pickle_in)

# Index page, Opens Automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'msg': 'Hi, Welcome to my API which is developed using FastAPI..!'}

@app.get('/{name}')
def get_name(name: str):
    return {'Hello, How Are You?': f'{name}'}

@app.post('/predict')
def predict_notes(data:BankNote):
    data = data.dict()

    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])

    if (prediction[0] > 0.5):
        prediction = "It is Fake Note"
    else:
        prediction = "It is Original Note"
    
    return {'prediction': prediction}

# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn app:app --reload