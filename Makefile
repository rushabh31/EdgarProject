SHELL:=/bin/bash
VENV_NAME:=edgar-test-venv
OUTLINES_PY := $(VENV_DIR)/lib/python3.10/site-packages/langchain_community/llms/outlines.py

## ----------------------------------------------------------------------
## Makefile with recipes
## ----------------------------------------------------------------------

help:	## Show help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

all: venv install_langchain_community modify_outlines

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
	pip install git+https://github.com/langchain-ai/langchain.git@master#subdirectory=libs/community \
	pre-commit install; \
	python -m ipykernel install --name $(VENV_NAME) --display-name $(VENV_NAME) --user;

# Modify the specific line in outlines.py
modify_outlines: install_langchain_community
	source $(VENV_NAME)/bin/activate; 
	sed -i.bak 's/self.client = models.transformers(self.model, **self.model_kwargs)/self.client = models.transformers(self.model, device='\''auto'\'', **self.model_kwargs)/' $(OUTLINES_PY)
