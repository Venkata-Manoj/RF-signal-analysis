.PHONY: install dev data test verify run clean

install:
	python -m pip install -r requirements.txt

dev:
	python -m pip install -r requirements.txt -r requirements-dev.txt

data:
	python scripts/generate_test_data.py

test:
	pytest

verify: dev data test

run:
	python scripts/run_app.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__
