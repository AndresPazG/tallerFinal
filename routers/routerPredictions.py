import pickle
from fastapi import APIRouter
import numpy as np
from interfaces import DataTest

router=APIRouter()
#Cargamos el modelo
with open("SVC_CropRecommendation.pkl","rb") as file:
    model=pickle.load(file)

@router.post("/predict")
def predict(data:DataTest):
    data=data.model_dump()
    print(data)
    
    #Array
    N=data["N"]
    P=data["P"]	
    K=data["K"]	
    temperature=data["temperature"]
    humidity=data["humidity"]	
    ph=data["ph"]	
    rainfall=data["rainfall"]
    
    xin = np.array([N,P,K,temperature,humidity,ph,rainfall]).reshape(1,7)
    prediction=model.predict(xin)

    print("predictions",prediction)
    return{"Prediction":prediction.tolist()}