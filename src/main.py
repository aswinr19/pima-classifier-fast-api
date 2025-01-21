from fastapi import FastAPI
from pydantic import BaseModel

class Data(BaseModel):
    no_of_times_pregnant: int = 0
    plasma_glucose_concentration: float = 0.0
    diastolic_blood_pressure: float = 0.0
    triceps_skin_fold_thickness: float = 0.0
    two_hour_serum_insulin: float = 0.0
    body_mass_index: float = 0.0
    diabetes_pedigree_function: float = 0.0
    age: int = 0


app = FastAPI()

@app.get("/")
def hello():
    return { "Hello" : "World!" }

@app.post("/check-diabetes")
