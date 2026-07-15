# Sandbox image for running model-generated tests.
# Build once:  docker build -t testloop-sandbox .
FROM python:3.12-slim

# Preinstall the test tooling so each run starts instantly and needs no network
# (containers run with --network=none).
RUN pip install --no-cache-dir pytest pytest-cov

WORKDIR /work
