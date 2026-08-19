.PHONY: test lint docs ci-docker clean

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pydoclint hbw/

docs:
	uv run sphinx-build -W docs docs/_build/html

ci-docker:
	docker run --rm -v $(PWD):/app -w /app python:3.12 bash -c "pip install uv && uv sync --all-groups && uv run pytest"

clean:
	rm -rf dist/ docs/_build/ .pytest_cache/ .ruff_cache/
