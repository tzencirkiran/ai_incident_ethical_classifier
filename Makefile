PYTHON := /home/talha/miniforge3/envs/pyt/bin/python -u
MODEL_DIR := model
SPLIT ?= random
PROCESSED_DIR ?= $(MODEL_DIR)/processed/$(SPLIT)
TUNING_CONFIG ?=
PREPROCESS_ARGS ?= $(if $(filter temporal,$(SPLIT)),--split temporal --val-year 2024 --test-year 2025,--split random)

HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: help preprocess baseline train tune test infer serve clean-processed clean-checkpoint clean

help:
	@echo "Targets:"
	@echo "  make preprocess [SPLIT=random] [ARGS=\"...\"] - build split artifacts under model/processed/\$${SPLIT}"
	@echo "  make baseline [SPLIT=random]                 - train/evaluate a TF-IDF logistic regression baseline"
	@echo "  make train [SPLIT=random] [ARGS=\"...\"]       - fine-tune bert-tiny and tune per-label thresholds"
	@echo "  make tune [SPLIT=random] [TUNING_CONFIG=...]  - run TinyBERT grid for the selected split"
	@echo "  make test [SPLIT=random]                     - evaluate the fine-tuned checkpoint"
	@echo "  make infer HEADLINE=\"...\" [PURPOSE=\"...\"] [TECHNOLOGY=\"...\"] [SECTOR=\"...\"]"
	@echo "                            - predict ethical-issue tags for an incident"
	@echo "  make serve [HOST=127.0.0.1] [PORT=8000]      - serve the FastAPI presentation UI"
	@echo "Examples:"
	@echo "  make preprocess"
	@echo "  make preprocess SPLIT=temporal ARGS=\"--split temporal --val-year 2024 --test-year 2025\""
	@echo "  make baseline SPLIT=temporal"
	@echo "  make tune SPLIT=temporal"
	@echo "  make clean-processed      - remove generated train/val/test splits"
	@echo "  make clean-checkpoint     - remove the fine-tuned checkpoint"
	@echo "  make clean                - remove both processed data and checkpoint"

preprocess:
	$(PYTHON) $(MODEL_DIR)/preprocessing.py --split-name "$(SPLIT)" $(PREPROCESS_ARGS) $(ARGS)

baseline:
	$(PYTHON) $(MODEL_DIR)/baseline.py --processed-dir "$(PROCESSED_DIR)" $(ARGS)

train:
	$(PYTHON) $(MODEL_DIR)/fine_tune.py --processed-dir "$(PROCESSED_DIR)" $(ARGS)

tune:
	$(PYTHON) $(MODEL_DIR)/tune.py --processed-dir "$(PROCESSED_DIR)" --run-name "$(SPLIT)" \
		$(if $(TUNING_CONFIG),--config "$(TUNING_CONFIG)") $(ARGS)

test:
	$(PYTHON) $(MODEL_DIR)/evaluate.py --processed-dir "$(PROCESSED_DIR)" $(ARGS)

infer:
	@if [ -z "$(HEADLINE)" ]; then \
		echo "Usage: make infer HEADLINE=\"...\" [PURPOSE=\"...\"] [TECHNOLOGY=\"...\"]"; \
		exit 1; \
	fi
	$(PYTHON) $(MODEL_DIR)/infer.py "$(HEADLINE)" \
		$(if $(PURPOSE),--purpose "$(PURPOSE)") \
		$(if $(TECHNOLOGY),--technology "$(TECHNOLOGY)") \
		$(if $(DEPLOYER),--deployer "$(DEPLOYER)") \
		$(if $(DEVELOPER),--developer "$(DEVELOPER)") \
		$(if $(SYSTEM_NAME),--system-name "$(SYSTEM_NAME)") \
		$(if $(NEWS_TRIGGER),--news-trigger "$(NEWS_TRIGGER)") \
		$(if $(JURISDICTION),--jurisdiction "$(JURISDICTION)") \
		$(if $(SECTOR),--sector "$(SECTOR)")

serve:
	$(PYTHON) -m uvicorn presentation_app:app --host "$(HOST)" --port "$(PORT)"

clean-processed:
	rm -rf $(MODEL_DIR)/processed

clean-checkpoint:
	rm -rf $(MODEL_DIR)/checkpoint

clean: clean-processed clean-checkpoint
