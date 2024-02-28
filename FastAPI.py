from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

def predict(person):
    model = joblib.load('gradient_boosting_regressor (1).pkl')
    X_test = np.array([[person.male, person.age, person.height, person.weight, person.duration, person.heart_rate, person.body_temp]])
    predictions = model.predict(X_test)
    return predictions

class Person:
    def __init__(self, male, age,height,weight, duration, heart_rate,body_temp):
        self.male = male
        self.age = age
        self.height =  height
        self.weight = weight
        self.duration = duration
        self.heart_rate = heart_rate
        self.body_temp = body_temp

class MyItem(BaseModel):
    name:str
    price:float
    ready:int

class PredictItem(BaseModel):
    male: int
    age: int
    height: float
    weight: float
    duration: float
    heart_rate: float
    body_temp: float

app = FastAPI()

@app.get("/")
async def home():
    return "hello"
@app.post("/submit")
async  def submit(personParam :PredictItem):
    person = Person(personParam.male, personParam.age, personParam.height, personParam.weight, personParam.duration, personParam.heart_rate, personParam.body_temp)
    calo =predict(person)

    return str(calo)