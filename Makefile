PYTHON := /home/talha/miniforge3/envs/pyt/bin/python -u
MODEL_DIR := model

.PHONY: help preprocess baseline train test infer clean-processed clean-checkpoint clean

help:
	@echo "Targets:"
	@echo "  make preprocess           - build train/val/test splits from data/incidents_data.xlsx"
	@echo "  make baseline             - train/evaluate a TF-IDF logistic regression baseline"
	@echo "  make train                - fine-tune bert-tiny and tune per-label thresholds"
	@echo "  make test                 - evaluate the fine-tuned checkpoint"
	@echo "  make infer HEADLINE=\"...\" [PURPOSE=\"...\"] [TECHNOLOGY=\"...\"] [SECTOR=\"...\"]"
	@echo "                            - predict ethical-issue tags for an incident"
	@echo "  make clean-processed      - remove generated train/val/test splits"
	@echo "  make clean-checkpoint     - remove the fine-tuned checkpoint"
	@echo "  make clean                - remove both processed data and checkpoint"

preprocess:
	$(PYTHON) $(MODEL_DIR)/preprocessing.py

baseline:
	$(PYTHON) $(MODEL_DIR)/baseline.py

train:
	$(PYTHON) $(MODEL_DIR)/fine_tune.py

test:
	$(PYTHON) $(MODEL_DIR)/evaluate.py

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

clean-processed:
	rm -rf $(MODEL_DIR)/processed

clean-checkpoint:
	rm -rf $(MODEL_DIR)/checkpoint

clean: clean-processed clean-checkpoint
