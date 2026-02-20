# ============================================================
# Makefile — PracticalQuiz1 ML Pipeline
# Author: Zohaib
# ============================================================

PYTHON = python

# Default target: run full pipeline
.PHONY: all preprocess train clean

all: preprocess train
	@echo "============================================"
	@echo " Full pipeline completed successfully."
	@echo "============================================"

preprocess:
	@echo "[MAKE] Running data preprocessing..."
	$(PYTHON) src/preprocess.py

train: preprocess
	@echo "[MAKE] Running model training..."
	$(PYTHON) src/train.py

clean:
	@echo "[MAKE] Cleaning processed data and models..."
	-rm -rf data/processed/*.csv
	-rm -rf models/*.pkl
	@echo "[MAKE] Clean complete."
