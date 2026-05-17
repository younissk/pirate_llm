"""Gradio chat playground for younissk/nanoBeard.

The model wasn't trained for multi-turn dialogue, so each user message is
sent fresh as a one-shot SFT-style instruction. History is shown but not fed back.
"""

import json

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from tokenizers import Tokenizer

from config import Config
from model import GPT

REPO = "younissk/nanoBeard"
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

cfg_dict = json.load(open(hf_hub_download(REPO, "config.json")))
cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
model = GPT(cfg).eval()
load_model(model, hf_hub_download(REPO, "model.safetensors"))
tok = Tokenizer.from_file(hf_hub_download(REPO, "pirate_bpe.json"))
EOS_ID = tok.token_to_id("<|endoftext|>")


@torch.no_grad()
def respond(message: str, history, max_new_tokens: int, temperature: float, top_k: int):
    prompt = PROMPT_TEMPLATE.format(instruction=message.strip())
    ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long)
    prompt_len = ids.size(1)

    for _ in range(int(max_new_tokens)):
        logits, _ = model(ids[:, -cfg.block_size :])
        logits = logits[:, -1] / max(float(temperature), 1e-5)
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

        if EOS_ID is not None and next_id.item() == EOS_ID:
            break

        # Stream the decoded response so far.
        yield tok.decode(ids[0, prompt_len:].tolist())

    yield tok.decode(ids[0, prompt_len:].tolist())


demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Slider(10, 300, value=120, step=10, label="Max new tokens"),
        gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature"),
        gr.Slider(0, 200, value=40, step=5, label="Top-k (0 = off)"),
    ],
    title="nanoBeard ☠️",
    description=(
        "Tiny pirate GPT (~14M params), trained on piratized TinyStories + "
        "Dolly-15k pirate SFT. Not trained for multi-turn chat — each message "
        "is treated as a fresh instruction. Outputs are decorative."
    ),
    examples=[
        ["Tell me a tale of buried treasure."],
        ["What does a pirate eat for breakfast?"],
        ["Write a short pirate poem."],
    ],
    cache_examples=False,
)


if __name__ == "__main__":
    demo.launch()
