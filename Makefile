.PHONY: help install env data tokenizer tokens dataset train train-gpu sample clean clean-data clean-ckpt

UV ?= uv
PROMPT ?= Once upon a time

help:
	@echo "Pirate-LLM make targets"
	@echo ""
	@echo "  make install       Install Python deps via uv"
	@echo "  make env           Copy example.env -> .env (won't overwrite existing)"
	@echo ""
	@echo "  make dataset       Full data pipeline: piratize + tokenizer + tokenize"
	@echo "  make data          Piratize TinyStories -> dataset/tiny_stories_pirate"
	@echo "  make tokenizer     Train BPE tokenizer  -> pirate_bpe.json"
	@echo "  make tokens        Tokenize corpus      -> train.bin, val.bin"
	@echo ""
	@echo "  make train         M1 smoke test (CPU/MPS, ~50 iters)"
	@echo "  make train-gpu     Full GPU run (CUDA, bf16, W&B + HF Hub sync)"
	@echo ""
	@echo "  make sample PROMPT='Ahoy'   Generate from out/ckpt.pt"
	@echo ""
	@echo "  make clean         Remove caches + wandb dir"
	@echo "  make clean-ckpt    Remove out/ckpt.pt"
	@echo "  make clean-data    Remove tokenized bins + tokenizer + piratized dataset"

install:
	$(UV) sync

env:
	@if [ -f .env ]; then \
		echo ".env already exists — leaving alone"; \
	else \
		cp example.env .env && echo "Created .env from example.env — fill in your tokens"; \
	fi

# ----- Data pipeline -----

dataset/tiny_stories_pirate:
	$(UV) run python -m dataset.tiny_stories

data: dataset/tiny_stories_pirate

pirate_bpe.json: dataset/tiny_stories_pirate
	$(UV) run python -m dataset.tokenize_ds

tokenizer: pirate_bpe.json

train.bin val.bin: pirate_bpe.json dataset/tiny_stories_pirate
	$(UV) run python -m dataset.tokenize_corpus

tokens: train.bin val.bin

dataset: data tokenizer tokens

# ----- Training -----

train: train.bin val.bin
	$(UV) run python -m training.train

train-gpu: train.bin val.bin
	$(UV) run python -c "from training.config import Config; from training.train import train; train(Config.for_gpu_training())"

# ----- Sampling -----

sample:
	$(UV) run python -m training.sample --prompt "$(PROMPT)"

# ----- Cleanup -----

clean:
	rm -rf wandb/ .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-ckpt:
	rm -rf out/

clean-data:
	rm -f train.bin val.bin pirate_bpe.json
	rm -rf dataset/tiny_stories_pirate
