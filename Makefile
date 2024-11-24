SHELL:=/bin/bash
VENV_NAME:=edgar-test-venv
OUTLINES_PY := $(VENV_NAME)/lib/python3.10/site-packages/langchain_community/llms/outlines.py

## ----------------------------------------------------------------------
## Makefile with recipes
## ----------------------------------------------------------------------

help:	## Show help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

all: venv modify_outlines

########################################################
# Local developement recipes
########################################################

venv:	## Create python environment, install pre-commit hooks, install dbt dps, and create .env

	@if [ -d "/opt/python/3.10.13" ]; then \
		echo "/opt/python/3.10.13 found. Building venv using this python."; \
		/opt/python/3.10.13/bin/python3 -m venv $(VENV_NAME); \
	else \
		echo "/opt/python/3.10.13 not found. Building venv using python3."; \
		python3 -m venv $(VENV_NAME); \
	fi
	source $(VENV_NAME)/bin/activate; \
	pip install wheel;\
	pip install -r requirements.txt; \
	pre-commit install; \
	python -m ipykernel install --name $(VENV_NAME) --display-name $(VENV_NAME) --user;

# Modify the specific line in outlines.py
modify_outlines:  venv ## Modify outlines.py.
	@source $(VENV_NAME)/bin/activate && \
	if [ -f "$(OUTLINES_PY)" ]; then \
		sed -i.bak "s/self.client = models.transformers(self.model, \*\*self.model_kwargs)/self.client = models.transformers(self.model, device='auto', \*\*self.model_kwargs)/" $(OUTLINES_PY); \
		echo "Modified $(OUTLINES_PY) successfully."; \
	else \
		echo "Error: $(OUTLINES_PY) not found."; \
		exit 1; \
	fi
