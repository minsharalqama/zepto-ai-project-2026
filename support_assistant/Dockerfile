FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5

COPY requirements.txt ./requirements.txt
COPY docker-constraints.txt ./docker-constraints.txt


RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.14.0+cpu"

RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --constraint docker-constraints.txt \
    -r requirements.txt

COPY . .


RUN python ingest.py

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
