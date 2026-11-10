# Run from Linux, macOS or WSL2 (native Windows on ARM64 cannot install the training stack).
SHELL := /bin/bash
.DEFAULT_GOAL := help

UV        ?= uv
RUN       := $(UV) run --no-sync
BENCHMARK ?= reference
IMAGE     ?= lstm-forecast-serve
TAG       ?= dev
MODEL_DIR ?= artifacts/model

.PHONY: help install lock data train evaluate benchmark test test-all lint format clean \
        data-eia train-production promote export-model smoke-bundle mlflow-ui \
        serve latency docker-build docker-run test-integration contract-snapshot

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install every extra plus dev tools into the uv environment
	$(UV) sync --all-extras

lock: ## Re-resolve uv.lock
	$(UV) lock

# ---------------------------------------------------------------- research benchmark

data: ## Download, preprocess, validate and snapshot the UCI benchmark dataset
	$(RUN) forecast data uci

train: data ## Fit every model of $(BENCHMARK) across its rolling-origin folds (logs to MLflow)
	$(RUN) forecast train --benchmark $(BENCHMARK) --skip-existing

evaluate: ## Score $(BENCHMARK) forecasts, write tables/plots, check vs reference
	$(RUN) forecast evaluate --benchmark $(BENCHMARK)

benchmark: ## Apples-to-apples benchmark: every model on the same 5 folds
	$(MAKE) train evaluate BENCHMARK=aligned

# ---------------------------------------------------------------- production model

data-eia: ## Fetch two years of EIA-930 demand (needs EIA_API_KEY) and snapshot it
	$(RUN) forecast data eia

train-production: ## Train, calibrate, export and register the production LSTM
	$(RUN) forecast train-production --config production

promote: ## Champion/challenger gate for the latest registered version
	$(RUN) forecast promote --config production

export-model: ## Download the production bundle into $(MODEL_DIR)
	$(RUN) forecast export --config production --out $(MODEL_DIR)

smoke-bundle: ## Tiny synthetic bundle into $(MODEL_DIR) (tests, CI images)
	$(RUN) forecast smoke-bundle --out $(MODEL_DIR)

mlflow-ui: ## Browse runs and the model registry at http://127.0.0.1:5000
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000

# ---------------------------------------------------------------- serving

serve: ## Run the API locally on :8080 (logs predictions to var/predictions)
	$(RUN) forecast serve --model-dir $(MODEL_DIR)

latency: ## PyTorch vs ONNX Runtime CPU latency -> experiments/results/latency.csv
	$(RUN) forecast bench-latency --bundle $(MODEL_DIR)

docker-build: ## Build the serving image around the bundle in $(MODEL_DIR)
	docker build -f docker/Dockerfile.serve --build-arg MODEL_DIR=$(MODEL_DIR) -t $(IMAGE):$(TAG) .

docker-run: ## Serve the image on :8080
	docker run --rm -p 8080:8080 --name forecast-api $(IMAGE):$(TAG)

test-integration: ## Black-box tests against a running container ($$FORECAST_API_URL)
	FORECAST_API_URL=$${FORECAST_API_URL:-http://localhost:8080} $(RUN) pytest -m integration -q

contract-snapshot: ## Accept an intentional API contract change
	$(RUN) forecast openapi --out tests/contract/openapi.snapshot.json

# ---------------------------------------------------------------- quality

test: ## Fast unit tests
	$(RUN) pytest -m "not slow and not integration" -q

test-all: ## All tests except container integration tests
	$(RUN) pytest -m "not integration" -q

lint: ## Ruff lint and format check
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

format: ## Apply ruff fixes and formatting
	$(RUN) ruff check --fix src tests
	$(RUN) ruff format src tests

clean: ## Remove caches (keeps data, forecasts and MLflow runs)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
