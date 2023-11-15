# Makefile for running Python scripts within a virtual environment 

# Set the name of the virtual environment 
VENV_NAME = CS15_2_virtual

# Define the virtual environment activation command 
ACTIVATE_VENV = source $(VENV_NAME)/bin/activate

# Define the Python interpreter to use
PYTHON = python

# Define the target to create and activate the virtual environment 
venv: 
	@echo "Creating virtual environment and installing dependencies ..."
	@python -m venv $(VENV_NAME)
	@$(ACTIVATE_VENV); \
	pip install -r requirements.txt 

# Defind the target to run the new script 
run_script: venv 
	@echo "Unzipping dataset..."
	@unzip -o CS15_2_virtual/DATASET/memotion_dataset_7k.zip -d CS15_2_virtual/DATASET/
	@unzip -o CS15_2_virtual/DATASET/MVSA.zip -d CS15_2_virtual/DATASET/
	@unzip -o CS15_2_virtual/DATASET/MVSA_single.zip -d CS15_2_virtual/DATASET/
	@echo "Running script.py within virtual environment ..."
	@$(ACTIVATE_VENV); \
	$(PYTHON) script.py; \
	deactivate
	@echo "Clean zip files..."
	@rm -rf CS15_2_virtual/DATASET/memotion_dataset_7k

# Define a clean target to remove any generated files or directories 
clean:
	@echo "Clean up ..." 
	@rm -rf __pycache__ # Remove compiled Python files
	@rm -rf .pytest_cache # Remove pytest cache (if any)
	@rm -f .coverage # Remove coverage data (if any)

.PHONY: venv run_script clean
