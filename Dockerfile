# syntax=docker/dockerfile:1.2
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3.9 python3.9-dev
COPY requirements.txt .
RUN --mount=type=cache,mode=0755,target=/root/.cache pip install -r requirements.txt
COPY . .
CMD ["python", "run.py"]