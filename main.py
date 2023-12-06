from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
app = FastAPI()

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def process_columns(df):
    # Убираем единицы измерения и кастуем к float
    df['mileage'] = df['mileage'].str.extract('(\d+\.\d+|\d+)').astype(float)
    df['engine'] = df['engine'].str.extract('(\d+)').astype(float)
    df['max_power'] = df['max_power'].str.extract('(\d+\.\d+|\d+)').astype(float)

    return df

def transform_data(data):
    data = process_columns(data)
    categorical_columns = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
    data['year_squared'] = data['year'] ** (2)
    data['preservation'] = 1 / (data['mileage'] + 1)
    data['mp_eng'] = data['max_power'] * data['engine']
    data['pow/vol'] = data['max_power'] / data['engine']
    data['km/year'] = data['km_driven'] / data['year']
    data['efficiency'] = data['km_driven'] * data['mileage'] * data['year'] * data['engine']
    data['transportation'] = data['max_power'] / data['seats']
    data['trans_val'] = np.where(data['transmission'] == 'Automatic', 100000, 200000)
    data['fuel_val'] = np.where(data['fuel'] == 'Diesel', 50000, 200000)
    data['benefit'] = data['fuel_val'] / (data['efficiency'] + 1)
    print(data.columns)
    return data.drop(['selling_price', 'torque'] + categorical_columns, axis=1)

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # Преобразование данных в формат, подходящий для вашей модели
    data = transform_data(pd.DataFrame([item.dict()]))  # преобразование одного объекта в DataFrame
    prediction = model.predict(data)  # предсказание стоимости машины
    return float(prediction[0])  # возвращаем предсказание в виде float

@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    # Преобразование данных в формат, подходящий для вашей модели
    data = pd.DataFrame([item.dict() for item in items.objects])  # преобразование списка объектов в DataFrame
    predictions = model.predict(data)  # предсказание стоимости машин
    return predictions.tolist()  # возвращаем предсказания в виде списка float