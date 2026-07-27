.DEFAULT_GOAL := help
PY ?= python
CONFIG ?= config/config.yaml

.PHONY: help install install-local install-api lint test ingest serve eval bench ablation clean \
        docker-up docker-down demo calibrate gate report update-baseline history selective dashboard

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install core package (mock mode works with just this)
	$(PY) -m pip install -e ".[dev,viz]"

install-local: ## Install local ML backends (sentence-transformers, faiss, transformers)
	$(PY) -m pip install -e ".[local,store,viz,dev]"

install-api: ## Install API backends (openai, anthropic)
	$(PY) -m pip install -e ".[api,store,viz,dev]"

lint: ## Ruff lint
	ruff check src eval tests

test: ## Run unit + integration tests (mock mode, no network)
	pytest

ingest: ## Build indexes from the corpus
	$(PY) -m arag.cli ingest --config $(CONFIG)

serve: ## Run the FastAPI server
	$(PY) -m arag.cli serve --config $(CONFIG)

eval: ## Run the eval harness for the current config over the gold set
	$(PY) -m arag.cli eval --config $(CONFIG)

ablation: ## Run the full ablation matrix and write RESULTS.md
	$(PY) -m eval.ablation $(CONFIG)

calibrate: ## Validate the critic/judge against the human-labeled set
	$(PY) -m eval.calibrate_judge $(CONFIG)

gate: ## Run the CI regression gate (diff vs committed baseline)
	$(PY) -m eval.ci_gate --config $(CONFIG)

report: ## Render the PR eval-report comment locally -> comment.md
	$(PY) -m eval.ci_gate --config $(CONFIG) --report comment.md --badge eval/results/badge.json

update-baseline: ## Re-baseline the regression gate (commit the result in the same PR)
	$(PY) -m eval.ci_gate --config $(CONFIG) --update-baseline

history: ## Show the experiment registry (recent eval runs)
	$(PY) -m eval.registry

selective: ## Risk–coverage analysis of the abstention gate
	$(PY) -m eval.selective $(CONFIG)

dashboard: ## Generate a self-contained HTML eval dashboard
	$(PY) -m eval.dashboard $(CONFIG)

bench: ablation ## Alias for ablation

demo: ## End-to-end smoke test in mock mode (ingest + a sample query)
	$(PY) -m arag.cli ingest --config $(CONFIG)
	$(PY) -m arag.cli query "How do I define a path parameter?" --config $(CONFIG)

docker-up: ## Start qdrant + redis + app
	docker compose up -d

docker-down: ## Stop the stack
	docker compose down

clean: ## Remove built indexes and caches
	rm -rf .arag_index eval/results/*.json .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
