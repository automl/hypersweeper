.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install check formatbump-version release
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-docs## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-docs: ## remove docs artifacts
	cd docs && make clean

ruff: ## run ruff as a formatter
	uvx ruff format hydra_plugins
	uvx ruff check --silent --exit-zero --no-cache --fix hydra_plugins
	uvx ruff check --exit-zero hydra_plugins
isort:
	uvx isort hydra_plugins tests

test: ## run tests quickly with the default Python
	python -m pytest tests
cov-report:
	coverage html -d coverage_html

coverage: ## check code coverage quickly with the default Python
	coverage run --source hydra_plugins -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/hypersweeper.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ hydra_plugins
	$(MAKE) -C docs html

bump-version: ## bump the version -- add current version number and kind of upgrade (minor, major, patch) as arguments
	bump-my-version bump --current-version

release: dist ## package and upload a release
	twine upload --repository testpypi dist/*
	@echo
	@echo "Test with the following:"
	@echo "* Create a new virtual environment to install the uploaded distribution into"
	@echo "* Run the following:"
	@echo
	@echo "        pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hypersweeper"
	@echo
	@echo "* Run this to make sure it can import correctly, plus whatever else you'd like to test:"
	@echo
	@echo "        python -c 'import hypersweeper'"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo
	@echo "    python -m twine upload dist/*"

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist
install: clean ## install the package to the active Python's site-packages
	uv pip install -e . --config-settings editable_mode=compat

install-dev: clean ## install the package to the active Python's site-packages
	uv pip install -e ".[dev,examples,doc,all]" --config-settings editable_mode=compat

check:
	uvx pre-commit run --all-files

format:
	make ruff
	make isort