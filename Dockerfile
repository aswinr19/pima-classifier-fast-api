FROM python:3.10-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src /code/src

COPY ./model/pima-classifier-model.pt /code/src

CMD ["fastapi", "run", "src/main.py", "--port", "80"]
