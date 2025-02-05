PYTHON := python3
SRC_DIR := src
MODELING_DIR := $(SRC_DIR)/modeling
DATA_DIR := data

# Source files
SRC_FILES := $(SRC_DIR)/dataset12_24.py $(SRC_DIR)/features.py $(SRC_DIR)/preProcess.py
MODELLING_FILES := $(MODELING_DIR)/predict.py $(MODELING_DIR)/train.py

.PHONY: all dataset preprocess features predict train clean

# Run the full pipeline
all: dataset preprocess features train predict

# Process dataset
dataset:
	$(PYTHON) $(SRC_DIR)/dataset12_24.py

# Preprocess data
preprocess: dataset
	$(PYTHON) $(SRC_DIR)/preProcess.py

# Extract features
features: preprocess
	$(PYTHON) $(SRC_DIR)/features.py

# Train the model
train: features
	$(PYTHON) $(MODELING_DIR)/train.py

# Run model predictions
predict: train
	$(PYTHON) $(MODELING_DIR)/predict.py

# Clean up temporary files
clean:
	rm -rf __pycache__ $(SRC_DIR)/**/__pycache__ $(DATA_DIR)/**/__pycache__ $(MODELING_DIR)/**/__pycache__
