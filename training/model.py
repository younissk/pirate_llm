import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.config import Config


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True,
        )

        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.triu(
            torch.full((config.block_size, config.block_size), float("-inf")),
            diagonal=1,
        )

        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        mask = self.causal_mask[:T, :T]

        y, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)

        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # horizontal mixing
        x = x + self.mlp(self.ln_2(x))  # vertical processing
        return x


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Embedding tables: one for token identity, one for position.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # position embeddings
        self.drop = nn.Dropout(config.dropout)

        # The stack of transformer blocks.
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final layer norm before the LM head.
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Language modeling head: project from n_embd back to vocab_size logits.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the input embedding matrix and the LM head share weights.
        # Both are shape (vocab_size, n_embd) — reuse saves vocab_size * n_embd
        # parameters. At our scale: 8192 * 384 ≈ 3.1M params saved (huge for tiny models).
        self.wte.weight = self.lm_head.weight

        # Initialize weights — small standard normal for embeddings/linears,
        # zero for biases, ones for LayerNorm. Matches GPT-2 / nanoGPT.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}; block_size is {self.config.block_size}"
        )

        # 1. Token embeddings: (B, T) -> (B, T, n_embd)
        tok_emb = self.wte(idx)

        # 2. Position embeddings: just 0, 1, 2, ..., T-1.
        # nn.Embedding lookup -> (T, n_embd), broadcasts over batch.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)

        # 3. Combine and apply input dropout.
        x = self.drop(tok_emb + pos_emb)

        # 4. Run through the stack of transformer blocks.
        for block in self.blocks:
            x = block(x)

        # 5. Final norm.
        x = self.ln_f(x)

        # 6. Project to vocabulary logits.
        logits = self.lm_head(x)

        # 7. Compute loss if targets given.
        loss = None
        if targets is not None:
            # Cross-entropy expects (N, vocab_size) and (N,) where N = B*T.
            # Flatten the batch and time dimensions together.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def num_parameters(self) -> int:
        """Count parameters. Excludes the position embedding by convention (small)."""
        n = sum(p.numel() for p in self.parameters())
        n -= self.wpe.weight.numel()
        return n
