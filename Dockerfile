FROM python:3.11-slim

# 安装 Tesseract OCR 及中文语言包
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
