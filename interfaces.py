from pydantic import BaseModel

class DataTest(BaseModel):
    N: float
    P: float	
    K: float 	
    temperature: float
    humidity:float	
    ph:float	
    rainfall:float