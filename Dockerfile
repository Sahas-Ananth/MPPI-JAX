FROM ghcr.io/nvidia/jax:jax-2025-02-19
ARG ENV=production
WORKDIR /app

COPY requirements.txt requirements-dev.txt ./

RUN if [ "$ENV" = "production" ]; then \
    pip install --no-cache-dir -r requirements.txt; \
    else \
    pip install --no-cache-dir -r requirements-dev.txt; \
    fi

COPY . ./

