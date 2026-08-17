"""ModernTransformer architecture, extracted from the training notebook at
~/PROJECTS/LLM_FROM_SCRATCH/PIPELINE/docker-model-scratch.ipynb (cell 11) so it can be
imported instead of redefined inline. Layer-for-layer identical to the trained model;
the only change is that `device` is a constructor parameter instead of a hardcoded
global, so the exact trained weights can be loaded and run on CPU.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

BLOCK_SIZE = 256
N_EMBD = 512
N_HEAD = 8
N_KV_HEAD = 2
N_LAYER = 20
VOCAB_SIZE = 50257


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta=10000.0):
    assert head_dim % 2 == 0
    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (theta_numerator / head_dim))
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_embeddings(x, freqs_cos, freqs_sin):
    freqs_cos = freqs_cos[:x.size(1)].unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin[:x.size(1)].unsqueeze(0).unsqueeze(2)

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    out = torch.empty_like(x)
    out[..., ::2] = x_even * freqs_cos - x_odd * freqs_sin
    out[..., 1::2] = x_even * freqs_sin + x_odd * freqs_cos
    return out


class GroupedQueryAttention(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.head_dim = N_EMBD // N_HEAD
        self.n_head = N_HEAD
        self.n_kv_head = N_KV_HEAD
        self.num_queries_per_kv = N_HEAD // N_KV_HEAD
        self.q_proj = nn.Linear(N_EMBD, N_HEAD * self.head_dim, bias=False)
        self.k_proj = nn.Linear(N_EMBD, N_KV_HEAD * self.head_dim, bias=False)
        self.v_proj = nn.Linear(N_EMBD, N_KV_HEAD * self.head_dim, bias=False)
        self.out_proj = nn.Linear(N_HEAD * self.head_dim, N_EMBD, bias=False)

        freqs_cos, freqs_sin = precompute_theta_pos_frequencies(self.head_dim, BLOCK_SIZE, device=device)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        q = apply_rotary_embeddings(q, self.freqs_cos, self.freqs_sin)
        k = apply_rotary_embeddings(k, self.freqs_cos, self.freqs_sin)
        k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(output)


class SwiGLUFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = int(2 * (4 * N_EMBD) / 3)
        self.w1 = nn.Linear(N_EMBD, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, N_EMBD, bias=False)
        self.w3 = nn.Linear(N_EMBD, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.attention = GroupedQueryAttention(device=device)
        self.feed_forward = SwiGLUFeedForward()
        self.attention_norm = RMSNorm(N_EMBD)
        self.ffn_norm = RMSNorm(N_EMBD)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class ModernTransformer(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([TransformerBlock(device=device) for _ in range(N_LAYER)])
        self.norm_final = RMSNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.token_embedding_table.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None):
        x = self.token_embedding_table(idx)
        for block in self.blocks:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
