# syntax=docker/dockerfile:1

FROM python:3.10.11-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

WORKDIR /app

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD exec gunicorn -w 4 app:app