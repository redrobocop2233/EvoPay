FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build dependencies needed by scientific Python packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# The repository intentionally excludes generated data/model binaries from Git.
# Create reproducible demo artifacts inside the deployment image instead.
RUN python -m data.generate_synthetic \
    && python -m models.train_pipeline

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 10000

CMD ["/app/start.sh"]
