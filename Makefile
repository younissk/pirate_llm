.PHONY: help install env data tokenizer tokens dataset train train-gpu sft sample publish publish-space test test-all test-fast test-slow clean clean-data clean-ckpt

UV ?= uv
CONFIG ?= sloop
CONFIG_FILE = configs/$(CONFIG).py
DATA_DIR = data/$(CONFIG)
PROMPT ?= Once upon a time

help:
	@echo "nanoBeard make targets — pass CONFIG=<name> (default: sloop)"
	@echo ""
	@echo "  make install                Install Python deps via uv"
	@echo "  make env                    Copy example.env -> .env"
	@echo ""
	@echo "  make dataset                Full data pipeline for CONFIG"
	@echo "  make data                   Piratize TinyStories  -> $(DATA_DIR)/tiny_stories_pirate"
	@echo "  make tokenizer              Train BPE             -> $(DATA_DIR)/pirate_bpe.json"
	@echo "  make tokens                 Tokenize corpus       -> $(DATA_DIR)/train.bin, val.bin"
	@echo ""
	@echo "  make train CONFIG=$(CONFIG) CONFIG_VARIANT=smoke|gpu"
	@echo "                              Train model (default: smoke variant)"
	@echo "  make sft                    SFT a pretrained ckpt"
	@echo "  make sample PROMPT='Ahoy'   Generate from runs/$(CONFIG)/ckpt.pt"
	@echo "  make eval                   Perplexity + sample gallery -> evals/results/<date>/$(CONFIG)/"
	@echo "  make eval-quick             Fast eval (20 batches, short samples)"
	@echo "  make publish                Push CONFIG ckpt to its HF model repo"
	@echo "  make publish-space          Push playground Space"
	@echo ""
	@echo "  make test                   Fast tests (excludes slow marker)"
	@echo "  make test-slow              Slow integration tests only"
	@echo "  make test-all               Everything"
	@echo ""
	@echo "  make clean                  Remove caches + wandb dir"
	@echo "  make clean-ckpt             Remove runs/$(CONFIG)/"
	@echo "  make clean-data             Remove $(DATA_DIR)/ contents"

install:
	$(UV) sync

env:
	@if [ -f .env ]; then \
		echo ".env already exists — leaving alone"; \
	else \
		cp example.env .env && echo "Created .env from example.env — fill in your tokens"; \
	fi

# ----- Data pipeline (per CONFIG) -----

data:
	$(UV) run python -m nanobeard.dataset_pipeline.tiny_stories --data-dir $(DATA_DIR)

tokenizer:
	$(UV) run python -m nanobeard.dataset_pipeline.tokenize_ds --data-dir $(DATA_DIR)

tokens:
	$(UV) run python -m nanobeard.dataset_pipeline.tokenize_corpus --data-dir $(DATA_DIR)

dataset: data tokenizer tokens

# ----- Training -----

train:
	$(UV) run python -m nanobeard.train --config $(CONFIG_FILE)

train-gpu:
	CONFIG_VARIANT=gpu $(UV) run python -m nanobeard.train --config $(CONFIG_FILE)

sft:
	CONFIG_VARIANT=sft $(UV) run python -m nanobeard.sft --config $(CONFIG_FILE)

# ----- Sampling -----

sample:
	$(UV) run python -m nanobeard.sample --config $(CONFIG_FILE) --prompt "$(PROMPT)"

# ----- Publishing -----

publish:
	$(UV) run python -m nanobeard.publish --config $(CONFIG_FILE)

publish-space:
	$(UV) run python scripts/publish_space.py

# ----- Dataset to/from HF Hub (skips 30-min piratize on remote machines) -----

push-data:
	$(UV) run python -m nanobeard.dataset_pipeline.push_to_hf --data-dir $(DATA_DIR)

# ----- Vast.ai -----

vast-launch:
	CONFIG=$(CONFIG) ./scripts/vast_launch.sh

vast-ssh:
	@INSTANCE=$$(cat .vast_instance 2>/dev/null) && vastai ssh-url $$INSTANCE

vast-logs:
	@INSTANCE=$$(cat .vast_instance 2>/dev/null) && vastai logs $$INSTANCE

vast-destroy:
	./scripts/vast_destroy.sh

# ----- Eval -----

eval:
	$(UV) run python -m nanobeard.eval.run --config $(CONFIG_FILE)

eval-quick:
	$(UV) run python -m nanobeard.eval.run --config $(CONFIG_FILE) --n-batches 20 --max-new-tokens 50

# ----- Tests -----

test: test-fast

test-fast:
	$(UV) run pytest

test-slow:
	$(UV) run pytest -m slow

test-all:
	$(UV) run pytest -m "slow or not slow"

# ----- Cleanup -----

clean:
	rm -rf wandb/ .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-ckpt:
	rm -rf runs/$(CONFIG)/

clean-data:
	rm -rf $(DATA_DIR)/train.bin $(DATA_DIR)/val.bin $(DATA_DIR)/pirate_bpe.json $(DATA_DIR)/tiny_stories_pirate
