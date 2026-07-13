FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY pip-cache/ /pip-cache/

# Install entirely offline — no network calls
RUN pip install --no-cache-dir --no-index --find-links=/pip-cache -r requirements.txt

COPY . /dashboard_backend/

ENV PYTHONPATH=/app
ENV MARTA_ROOT=/app

WORKDIR /dashboard_backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]