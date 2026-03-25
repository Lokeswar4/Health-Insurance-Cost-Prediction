.PHONY: install lint format test run clean

install:
	uv sync --dev

lint:
	uv run ruff check src/ tests/ main.py

format:
	uv run ruff format src/ tests/ main.py

test:
	uv run pytest tests/ -v

run:
	uv run python main.py

run-model:
	uv run python main.py --model-only

clean:
	rm -rf outputs/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
