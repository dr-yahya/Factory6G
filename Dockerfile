FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

# System deps for matplotlib non-interactive rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV MPLBACKEND=Agg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/       src/
COPY models/    models/
COPY config/    config/
COPY data/      data/
COPY main.py    .
COPY config.json .

ENTRYPOINT ["python", "main.py"]
CMD ["--config", "config.json"]
