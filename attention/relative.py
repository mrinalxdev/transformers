import torch
import torch.nn as nn
import math

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_relative_position=32, dropout=0.0):
        super().__init__()

        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_relative_position = max_relative_position
        self.dropout = dropout



        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)


        # rpe embeddings
        self.relative_position_k = nn.Embedding(2 * max_relative_position + 1, self.d_k)
        self.relative_position_v = nn.Embedding(2 * max_relative_position + 1, self.d_k)


    def forward(self, x, mask=None):
        B, T, _ = x.size()
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)


        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # working with logits
        relative_position = self._get_relative_positions(T, x.device)

        rel_keys = self.relative_position_k(relative_position)
        rel_logits = torch.einsum("bhtd, ttd -> bhtt", q, rel_keys) / math.sqrt(self.d_k)

        scores = scores + rel_logits

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))


        

        attn = torch.softmax(scores, dim=-1)
        attn = torch.dropout(attn, p=self.dropout, train=self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)
    

    def _get_relative_positions(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        rel_pos = positions[:, None] - positions[None, :]
        rel_pos = torch.clamp(rel_pos, -self.max_relative_position, self.max_relative_position)

        rel_pos = rel_pos + self.max_relative_position

        return rel_pos






