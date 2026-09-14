FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --no-deps kaggle-environments==1.32.7 && pip install --no-cache-dir numpy scipy jsonschema requests
COPY inside.py /opt/arena/inside.py
ENV PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
USER 65534:65534
WORKDIR /work
