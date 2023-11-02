FROM python:3.9

WORKDIR /code

COPY /requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade pip 

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./ /code/app

CMD [ "uvicorn","app.main:app"]

EXPOSE 80