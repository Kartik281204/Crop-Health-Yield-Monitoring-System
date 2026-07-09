# Crop Disease Detection -- API image.
#
# NOTE: written and syntax-reviewed carefully, but not build-tested --
# this sandbox doesn't have Docker available. Everything it wraps (the
# model, the API, the test suite) *has* been run and verified for real;
# see the README for what was actually executed vs. just written.

FROM python:3.12-slim

WORKDIR /app

# libgomp1: TensorFlow's CPU kernels use OpenMP for multi-threading and
# expect it to be present; python:slim doesn't include it by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# src/ holds the model-loading + prediction logic that api/ imports from --
# both need to be copied, in this relative layout, for those imports to resolve.
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

WORKDIR /app/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
