# TODO
# 2) Add a ui for taking test data for prediction
# 4) Separate the model inference code into another file 
# 5) Remove the hardcoded values and make them dynamic
# 6) Fix the docker container

from operator import delitem
import os
import torch
import numpy as np 
from typing import Annotated
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Response, Request, Depends, Query, HTTPException
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sklearn.preprocessing import StandardScaler


class Data(SQLModel, table = True):
    id: int | None = Field(default=None, primary_key=True)
    no_of_times_pregnant: int | None = Field(default=None)
    plasma_glucose_concentration: float | None = Field(default=None)
    diastolic_blood_pressure: float | None = Field(default=None)
    triceps_skin_fold_thickness: float | None = Field(default=None)
    two_hour_serum_insulin: float | None = Field(default=None)
    body_mass_index: float | None = Field(default=None)
    diabetes_pedigree_function: float | None = Field(default=None) 
    age: int | None = Field(default=None, index=True)
    result: float | None = Field(default=None)

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

#def create_db_and_tables():
#    SQLModel.metadata.create_all(engine)

def create_db_and_tables():
    print("Creating database tables...")
    try:
        SQLModel.metadata.create_all(engine)
        print("Tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")


def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

def preprocess_data(x: list) -> list:

    scaler = StandardScaler()
    x = scaler.fit_transform(np.array(x).reshape(1,-1))

    return x


def diabetes_prediction_neural_net(x: list) -> dict:

    x = preprocess_data(x)

    tnsr = torch.tensor(x, dtype=torch.float32)

    model = torch.load('/root/documents/pima-classifier-fast-api/model/pima-classifier-model.pt')
    pred = model(tnsr)

    print(pred.item())
    
    return round(pred.item())


model_version = "0.1.0"

ml_models = {}
mean = np.empty((1,8))
std = np.empty((1,8))


@asynccontextmanager
async def lifespan(app: FastAPI):

    if not os.path.exists(sqlite_file_name):
        create_db_and_tables()

    mean = np.loadtxt('/root/documents/pima-classifier-fast-api/data/mean.csv')
    std = np.loadtxt('/root/documents/pima-classifier-fast-api/data/std.csv')

    ml_models["diabetes_nn"] = diabetes_prediction_neural_net
    yield

    ml_models.clear()
app = FastAPI(lifespan=lifespan)


app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

#@app.on_event("startup")
#def on_startup():
#    if not os.path.exists(sqlite_file_name):
#        print("application is starting...")
#        create_db_and_tables()
#

@app.get("/")
def hello():
    return { "helth_check" : "OK", "model_version": model_version }

@app.get("/check-diabetes", response_class=HTMLResponse)
async def predict_diabetes_page(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')


@app.post("/check-diabetes")
async def predict_diabetes(data: Data, session: SessionDep) -> Data:

    mean = np.loadtxt('/root/documents/pima-classifier-fast-api/data/mean.csv')
    std = np.loadtxt('/root/documents/pima-classifier-fast-api/data/std.csv')

    print("here")
    print(data.no_of_times_pregnant)
    print(data.plasma_glucose_concentration)
    print(data.diastolic_blood_pressure)
    print(data.triceps_skin_fold_thickness)
    print(data.two_hour_serum_insulin)
    print(data.body_mass_index)
    print(data.diabetes_pedigree_function)
    print(data.age)

    arr = [
        (float(data.no_of_times_pregnant) - mean[0]) / std[0],
        (float(data.plasma_glucose_concentration) - mean[1])/ std[1],
        (float(data.diastolic_blood_pressure) - mean[2]) / std[2],
        (float(data.triceps_skin_fold_thickness) - mean[3]) / std[3],
        (float(data.two_hour_serum_insulin) - mean[4]) / std[4],
        (float(data.body_mass_index) - mean[5]) / std[5],
        (float(data.diabetes_pedigree_function) - mean[6]) / std[6],
        (float(data.age) - mean[7]) / std[7],
    ]
    
    print(arr)
    result = ml_models["diabetes_nn"](arr)
    data.result = result

    try:
        session.add(data)
        session.commit()
        session.refresh(data)
    except Exception as e:
        print(f"Error saving data to database: {e}")
        raise HTTPException(status_code=500, detail="Error saving data to database.")

    return data 


@app.get("/get-data")
async def get_previous_data(session: SessionDep):
    data = session.exec(select(Data).offset(0).limit(10)).all()

    return data

