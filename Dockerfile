FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🛠️ HARDCORE FIX: Explicitly force install problematic packages directly in the build sequence
RUN pip install --no-cache-dir langchain-groq langchain-google-genai

COPY . .

EXPOSE 8080

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
