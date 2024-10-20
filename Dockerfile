FROM python:3.9.19-slim AS stg1

COPY requirements.txt .

RUN apt-get update -y && apt install -y --no-install-recommends git\
&& pip install --no-cache-dir -U pip \ 
    && pip install --user --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

# Stage 2: run application code
FROM python:3.9.19-slim

COPY --from=stg1 /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

WORKDIR  /opt/mount/
COPY . .


# ENTRYPOINT ["/bin/bash"]
# CMD ["python3", "src/train.py"]


### Using UV START
# # Build stage
# FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
# ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# WORKDIR /app

# # Install dependencies
# RUN --mount=type=cache,target=/root/.cache/uv \
# 	--mount=type=bind,source=uv.lock,target=uv.lock \
# 	--mount=type=bind,source=pyproject.toml,target=pyproject.toml \
# 	uv sync --frozen --no-install-project --no-dev

# # Copy the rest of the application
# ADD . /app

# # Install the project and its dependencies
# RUN --mount=type=cache,target=/root/.cache/uv \
# 	uv sync --frozen --no-dev

# # Final stage
# FROM python:3.12-slim-bookworm

# # Copy the application from the builder
# COPY --from=builder --chown=app:app /app /app

# # Place executables in the environment at the front of the path
# ENV PATH="/app/.venv/bin:$PATH"

# # Set the working directory
# WORKDIR /app


### Using UV END