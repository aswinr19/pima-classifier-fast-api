# TODO
# 1) Add life span manager for model
# 2) Add a ui for taking test data for prediction
# 3) Process the data before passing it to the model
# 4) Separate the model inference code into another sqlite_file_name
# 5) Remove the hardcoded values and make them dynamic
# 6) Fix the docker container


from contextlib import asynccontextmanager
import torch
import numpy as np 
from typing import Annotated
from fastapi import FastAPI, Depends, Query, HTTPException
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

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

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

    if pred.item() < 0.5:
        return { "prediction": pred.item(), "result": "You have tested -ve for diabetes"}
    else:
        return { "prediction": pred.item(), "result": "You have tested +ve for diabetes"}

model_version = "0.1.0"

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["diabetes_nn"] = diabetes_prediction_neural_net
    yield

    ml_models.clear()

app = FastAPI(lifespan=lifespan)


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/")
def hello():
    return { "helth_check" : "OK", "model_version": model_version }

@app.post("/check-diabetes")
async def predict_diabetes(data: Data, session: SessionDep) -> dict:
   
    arr = [
        data.no_of_times_pregnant,
        data.plasma_glucose_concentration,
        data.diastolic_blood_pressure,
        data.triceps_skin_fold_thickness,
        data.two_hour_serum_insulin,
        data.body_mass_index,
        data.diabetes_pedigree_function,
        data.age,
    ]

    #for elm in data:
    #    arr.append(elm[1])
    #    print(elm)

    #arr.pop(0)
    #arr.pop(-1)

    result = ml_models["diabetes_nn"](arr)
    data.result = result["prediction"]
    #copy = dict(data)

    #print(result["prediction"])
    #copy["result"] = result["prediction"]

    #print("updated copy")
    #print(copy)

    #copy = Data(**copy)

    print(data)

    session.add(data)
    session.commit()
    session.refresh(data)

    return result


#@app.get("/get-data")
#async def get_previous_data(session: SessionDep) -> list[Data]:
#    data = session.exec(select(Data).offset(0).limit(10)).all()
#
#    return data
