# ============================================================
# Makefile — PracticalQuiz1 ML Pipeline
# Author: Zohaib
# ============================================================

PYTHON = python

.PHONY: all preprocess train evaluate clean

# Run full pipeline sequentially
all: preprocess train evaluate
	@echo "============================================"
	@echo " Full ML pipeline completed successfully."
	@echo "============================================"

preprocess:
	@echo "[MAKE] Running data preprocessing..."
	$(PYTHON) src/preprocess.py

train:
	@echo "[MAKE] Running model training..."
	$(PYTHON) src/train.py

evaluate:
	@echo "[MAKE] Running model evaluation..."
	$(PYTHON) src/evaluate.py

clean:
	@echo "[MAKE] Cleaning generated files..."
	-rm -rf data/processed/*.csv
	-rm -rf models/*.pkl
	-rm -rf results/metrics.txt
	@echo "[MAKE] Clean complete."
