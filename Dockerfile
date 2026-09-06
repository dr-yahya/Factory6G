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
# sionna-rt==1.2.2 pins mitsuba==3.8.0 and drjit==1.3.1 itself, which are the
# versions this image used to force with `uv --override` because sionna-rt==1.2.1
# pinned mitsuba 3.7.1 with no arm64 wheel. The override is no longer needed.
RUN uv pip install --system --no-cache -r requirements.txt

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
