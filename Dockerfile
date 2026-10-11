# Stage 1: compile the C++ index into a Python extension.
FROM python:3.11-slim AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "cmake>=3.16" "pybind11>=2.12"
WORKDIR /app
COPY cpp cpp
# The CMake target writes the module to ../src/recsys relative to cpp/.
RUN mkdir -p src/recsys \
    && cmake -S cpp -B build/cpp -DHNSW_BUILD_PYTHON=ON \
             -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build build/cpp -j --target _hnsw

# Stage 2: the service, without compilers.
FROM python:3.11-slim
# LightGBM's wheel needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Dependencies first, so editing the source does not reinstall them.
COPY pyproject.toml .
RUN mkdir -p src/recsys && touch src/recsys/__init__.py \
    && pip install --no-cache-dir -e ".[serve]"
COPY src src
COPY --from=build /app/src/recsys/ src/recsys/

RUN useradd --create-home recsys
USER recsys
# Embeddings, model and serving files are mounted here, not baked in.
ENV RECSYS_ARTIFACTS=/artifacts
EXPOSE 8000
CMD ["uvicorn", "recsys.service:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
