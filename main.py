import io

import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.openapi.models import Response
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from starlette.responses import StreamingResponse

app = FastAPI()

df_train = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_test.csv')
df_unique_train = df_train.drop_duplicates(subset=df_train.columns.drop('selling_price'))
df_unique_test = df_test.drop_duplicates(subset=df_train.columns.drop('selling_price'))
df_unique_train = df_unique_train.reset_index(drop=True)
df_unique_test = df_unique_test.reset_index(drop=True)

model = LinearRegression()
random.seed(42)
np.random.seed(42)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
class Item(BaseModel):
    name: str
    year: int
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


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.__dict__])
    df.drop('name', axis=1, inplace=True)
    df.drop('fuel', axis=1, inplace=True)
    df.drop('seller_type', axis=1, inplace=True)
    df.drop('transmission', axis=1, inplace=True)
    df.drop('owner', axis=1, inplace=True)
    df.drop('torque', axis=1, inplace=True)
    y_test_pred = model.predict(df)
    return y_test_pred


@app.post("/predict_items")
async def upload_items_csv (file: UploadFile):
    # Сохранение загруженного файла
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)
    print("Успешно загружен файл")
    df = pd.read_csv(file.filename)
    df.drop('name', axis=1, inplace=True)
    df.drop('fuel', axis=1, inplace=True)
    df.drop('seller_type', axis=1, inplace=True)
    df.drop('transmission', axis=1, inplace=True)
    df.drop('owner', axis=1, inplace=True)
    df.drop('torque', axis=1, inplace=True)
    df['mileage'] = df['mileage'].astype(str).str.replace(' kmpl', '').str.replace(' km/kg','').astype( float)
    df['engine'] = df['engine'].astype(str).str.replace(' CC', '').astype(float)
    df['max_power'] = df['max_power'] = pd.to_numeric(df['max_power'].astype(str).str.replace(' bhp', ''), errors='coerce')
    print(df)
    predictions = model.predict(df)
    df["predictions"] = predictions
    output = io.StringIO()
    df.to_csv(output, index=False)
    print("подготовка ответа",output.getvalue())
    response = StreamingResponse(iter([output.getvalue()]),media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    return response

@app.on_event('startup')
async def startup_event():
    create_model()
    print("Сервис запущен! модель обучена")


def create_model():
    df_unique_train['mileage'] = df_unique_train['mileage'].astype(str).str.replace(' kmpl', '').str.replace(' km/kg',
                                                                                                             '').astype(
        float)
    df_unique_train['engine'] = df_unique_train['engine'].astype(str).str.replace(' CC', '').astype(float)

    df_unique_train['max_power'] = df_unique_train['max_power'] = pd.to_numeric(
        df_unique_train['max_power'].astype(str).str.replace(' bhp', ''), errors='coerce')

    mean_max_power = df_unique_train['max_power'].mean()

    df_unique_train['max_power'] = df_unique_train['max_power'].fillna(mean_max_power)

    df_unique_test['mileage'] = df_unique_test['mileage'].astype(str).str.replace(' kmpl', '').str.replace(' km/kg',
                                                                                                           '').astype(
        float)

    df_unique_test['engine'] = df_unique_test['engine'].astype(str).str.replace(' CC', '').astype(float)

    df_unique_test['max_power'] = df_unique_test['max_power'] = pd.to_numeric(
        df_unique_test['max_power'].astype(str).str.replace(' bhp', ''), errors='coerce')

    mean_max_power = df_unique_test['max_power'].mean()

    df_unique_test['max_power'] = df_unique_test['max_power'].fillna(mean_max_power)

    mileage_median = df_unique_train['mileage'].median()
    engine_median = df_unique_train['engine'].median()
    max_power_median = df_unique_train['max_power'].median()
    seats_median = df_unique_train['seats'].median()

    df_unique_train['mileage'] = df_unique_train['mileage'].fillna(mileage_median)
    df_unique_train['engine'] = df_unique_train['engine'].fillna(engine_median)
    df_unique_train['max_power'] = df_unique_train['max_power'].fillna(max_power_median)
    df_unique_train['seats'] = df_unique_train['seats'].fillna(seats_median)

    df_unique_test['mileage'] = df_unique_test['mileage'].fillna(mileage_median)
    df_unique_test['engine'] = df_unique_test['engine'].fillna(engine_median)
    df_unique_test['max_power'] = df_unique_test['max_power'].fillna(max_power_median)
    df_unique_test['seats'] = df_unique_test['seats'].fillna(seats_median)

    df_unique_train['engine'] = df_unique_train['engine'].astype(int)
    df_unique_train['seats'] = df_unique_train['seats'].astype(int)

    df_unique_test['engine'] = df_unique_test['engine'].astype(int)
    df_unique_test['seats'] = df_unique_test['seats'].astype(int)

    y_train = df_unique_train['selling_price']

    df_unique_train.drop('selling_price', axis=1, inplace=True)
    df_unique_train.drop('name', axis=1, inplace=True)
    df_unique_train.drop('seller_type', axis=1, inplace=True)
    df_unique_train.drop('fuel', axis=1, inplace=True)
    df_unique_train.drop('transmission', axis=1, inplace=True)
    df_unique_train.drop('owner', axis=1, inplace=True)
    df_unique_train.drop('torque', axis=1, inplace=True)

    X_train = df_unique_train

    df_unique_test.drop('name', axis=1, inplace=True)
    df_unique_test.drop('fuel', axis=1, inplace=True)
    df_unique_test.drop('seller_type', axis=1, inplace=True)
    df_unique_test.drop('transmission', axis=1, inplace=True)
    df_unique_test.drop('owner', axis=1, inplace=True)
    df_unique_test.drop('torque', axis=1, inplace=True)
    df_unique_test.drop('selling_price', axis=1, inplace=True)
    model.fit(X_train, y_train)