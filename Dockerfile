# syntax=docker/dockerfile:1

FROM python:3.8-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

WORKDIR /app

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENV HOST 0.0.0.0
ENV PORT 80

CMD ["gunicorn", "-w", "2", "-b" , "$HOST:$PORT", "app:app"]