FROM python:3.11-slim

WORKDIR /app

# System deps for matplotlib non-interactive rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV MPLBACKEND=Agg

RUN pip install --no-cache-dir uv

COPY requirements.txt .
# Use uv --override to force mitsuba==3.8.0 (only arm64 wheel available;
# sionna-rt==1.2.1 pins 3.7.1 which has no arm64 build)
RUN printf "mitsuba==3.8.0\ndrjit==1.3.1\n" > /tmp/overrides.txt && \
    uv pip install --system --no-cache -r requirements.txt --override /tmp/overrides.txt

COPY pyproject.toml .
COPY src/       src/
COPY models/    models/
COPY config/    config/
COPY data/      data/
COPY ["reference/dr_athirah_simulation/", "reference/dr_athirah_simulation/"]

# Editable install wires up the `factory6g` package + console entrypoints.
RUN uv pip install --system --no-cache -e .

ENTRYPOINT ["python", "-m", "factory6g.cli.run"]
CMD ["--config", "config/config.json"]
