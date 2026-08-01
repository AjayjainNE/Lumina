FROM python:3.11-slim
LABEL description="LUMINA Document Intelligence Platform"
RUN apt-get update && apt-get install -y build-essential libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY config/ ./config/
COPY .env.example .env
RUN mkdir -p outputs/audio outputs/eval checkpoints logs
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health || exit 1
ENV PYTHONPATH=/app/src
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
