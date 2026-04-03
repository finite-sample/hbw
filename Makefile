.PHONY: test lint docs ci-docker clean

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run mypy hbw/
	uv run pydoclint hbw/
	uv run vulture hbw/

docs:
	uv run sphinx-build docs docs/_build/html

ci-docker:
	docker run --rm -v $(PWD):/app -w /app python:3.12 bash -c "pip install uv && uv sync --dev && uv run pytest"

clean:
	rm -rf dist/ docs/_build/ .pytest_cache/ .mypy_cache/ .ruff_cache/
