from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import random
import json
from fastapi.responses import JSONResponse
import random

def convert_numpy(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy(item) for item in data]
    return data

def dailyCalories(gioi_tinh, tuoi, chieu_cao, can_nang):
    if gioi_tinh == 1:  # Nam
        daily_calories = 88.362 + (13.397 * can_nang) + (4.799 * chieu_cao) - (5.677 * tuoi)
    elif gioi_tinh == 0:  # Nữ
        daily_calories = 447.593 + (9.247 * can_nang) + (3.098 * chieu_cao) - (4.330 * tuoi)
    else:
        raise ValueError("Giới tính không hợp lệ. Vui lòng chỉ định là 1 hoặc 0.")
    return daily_calories

def tao_thuc_don(daily_calories):
    thuc_don = {}
    # Số ngày
    for i in range(1, 8):
        bua_sang = [("Bánh mì ngũ cốc", 0.2), ("Trứng bỏ lò", 0.1), ("Hành tây", 0.05), ("Dưa chuột", 0.05), ("Quả chuối", 0.1), ("Quả lê", 0.1), ("Cháo yến mạch", 0.15), ("Hạnh nhân", 0.05), ("Trái cây", 0.2)]
        bua_trua = [("Salad gà với sốt vinaigrette", 0.25), ("Canh bí đỏ với thịt gà và cà rốt", 0.35), ("Canh cải bắp với thịt gà và cà rốt", 0.35), ("Salad rau cải với đậu hủ chiên và sốt mù tạt", 0.25), ("Canh rau cải thảo với thịt gà và cà rốt", 0.35), ("Salad cà chua bi với thịt gà và hành tây", 0.25), ("Canh rau cải thảo với thịt gà và cà rốt", 0.35)]
        bua_toi = [("Cơm gạo lứt", 0.2), ("Cá hồi nướng", 0.4), ("Rau xà lách", 0.1), ("Thịt bò xào cải ngọt", 0.4), ("Cá basa nướng", 0.4), ("Thịt gà nướng", 0.4), ("Thịt bò xào rau cải", 0.4), ("Cá hồi nướng", 0.4), ("Rau cải xanh", 0.1)]
        
        thuc_don[f"Ngày {i}"] = {
            "Bữa sáng": random.sample(bua_sang, 3),
            "Bữa trưa": random.sample(bua_trua, 3),
            "Bữa tối": random.sample(bua_toi, 3)
        }
    return thuc_don

def in_menu(thuc_don, daily_calories, calo_burn):
    breakfast_ratio = 0.2
    lunch_ratio = 0.4
    dinner_ratio = 0.4
    menu_json = {
        "daily_calories": daily_calories,
        "calo_burn": calo_burn,
        "menu": {}
    }
    
    # Tính lượng calo cho mỗi bữa ăn
    calo_breakfast = daily_calories * breakfast_ratio
    calo_lunch = daily_calories * lunch_ratio
    calo_dinner = daily_calories * dinner_ratio

    for ngay, bua_an in thuc_don.items():
        bua_json = {}
        for bua, mon_an in bua_an.items():
            mon_an_json = []
            if bua == "Bữa sáng":
                calo_per_dish = calo_breakfast / 3
            elif bua == "Bữa trưa":
                calo_per_dish = calo_lunch / 3
            else:
                calo_per_dish = calo_dinner / 3

            for item in mon_an:
                ten_mon_an = item[0]
                mon_an_json.append({
                    "món": ten_mon_an,
                    "khối_lượng_calo": float(calo_per_dish),
                    "khối_lượng_thức_ăn": round(calo_per_dish / 4, 2)
                })
            bua_json[bua] = mon_an_json
        menu_json["menu"][ngay] = bua_json

    menu_json = convert_numpy(menu_json)
    return menu_json

def main():
    tuoi = int(input("Nhập tuổi của bạn: "))
    chieu_cao = float(input("Nhập chiều cao của bạn (cm): "))
    can_nang = float(input("Nhập cân nặng của bạn (kg): "))
    gioi_tinh = int(input("Nhập giới tính của bạn (1 cho nam, 0 cho nữ): "))

    # Tính toán nhu cầu calo hàng ngày
    if gioi_tinh == 1 or gioi_tinh == 0:
        daily_calories = dailyCalories(gioi_tinh, tuoi, chieu_cao, can_nang)
        print("Nhu cầu calo hàng ngày của bạn là:", daily_calories)
        
        # Tạo và in ra thực đơn
        thuc_don = tao_thuc_don(daily_calories)
        menu_json_str = in_menu(thuc_don, daily_calories,1)
        print(menu_json_str)
    else:
        print("Giới tính không hợp lệ. Vui lòng chỉ định là 1 hoặc 0.")


if __name__ == "__main__":
    main()

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
    json_content = {"key": "value",
                    "key1": "value"}
    return JSONResponse(content=json_content)
@app.post("/submit")
async  def submit(personParam :PredictItem):
    person = Person(personParam.male, personParam.age, personParam.height, personParam.weight, personParam.duration, personParam.heart_rate, personParam.body_temp)
    calo =predict(person)

    if personParam.male == 1 or personParam.male == 0:
        daily_calories = dailyCalories(personParam.male, personParam.age, personParam.height, personParam.weight)        

        thuc_don = convert_numpy(tao_thuc_don(daily_calories))
        menu_json_str = in_menu(thuc_don, daily_calories,calo)
        return menu_json_str
    else:
        return None
