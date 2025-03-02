.PHONY: clean lint requirements dev-setup

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "*.DS_Store" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +

lint:
	flake8 src
	black src --check

format:
	black src

test:
	python -m pytest tests/

requirements:
	pip install -r requirements.txt

dev-setup: requirements
	pip install -e .
	pre-commit install

run:
	python src/main.py 