FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV ENV=production
# Cloud Run sets PORT itself at runtime; this is just a sane local default.
ENV PORT=8080

CMD ["python", "main.py"]
