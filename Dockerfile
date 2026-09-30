# juris-rag inference container (Lambda container image / Fargate compatible)
FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

# HF model cache is baked at build time so cold starts skip downloads
ENV HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    PYTHONUNBUFFERED=1 \
    PORT=8080

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Pre-download the bi-encoder + cross-encoder into the image layer.
# (Runs once at build; keeps cold starts to model-load only.)
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY src/ src/
COPY data/corpus/ data/corpus/
COPY service.py .

# Index is built at first /index call or container start; corpus is bundled.
CMD ["sh", "-c", "uvicorn service:app --host 0.0.0.0 --port ${PORT}"]
