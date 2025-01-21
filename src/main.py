import torch
import numpy as np 
from typing import Annotated
from fastapi import FastAPI, Depends, Query, HTTPException
from sqlmodel import Field, Session, SQLModel, create_engine, select


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


app = FastAPI()


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/")
def hello():
    return { "Hello" : "World!" }

@app.post("/check-diabetes")
#async def predict_diabetes(data: Data, session: SessionDep) -> Data:
async def predict_diabetes(data: Data, session: SessionDep):
    session.add(data)
    session.commit()
    session.refresh(data)

    arr = []

    for elm in data:
        arr.append(elm[1])
        print(elm)
    
    arr.pop(0)

    tnsr = torch.tensor(arr)

    model = torch.load('/root/documents/pima-classifier-fast-api/model/pima-classifier-model.pt')
   
    pred = model(tnsr)

    if pred == 0:
        return { "result": "You have tested -ve for diabetes"}
    else:
        return { "result": "You have tested +ve for diabetes"}
    #return data


#@app.get("/get-data")
#async def get_previous_data(session: SessionDep) -> list[Data]:
#    data = session.exec(select(Data).offset(0).limit(10)).all()
#
#    return data
