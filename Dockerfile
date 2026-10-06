FROM python:3.12-slim-bookworm

# CPU image on purpose. Do not switch the base to nvidia/cuda or install the PyPI torch wheel.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY assets ./assets

RUN pip install --no-cache-dir 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

CMD ["python", "-c", "import torch; assert torch.version.cuda is None; print(torch.__version__)"]
