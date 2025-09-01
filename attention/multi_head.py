import math
import torch.nn as nn

from .attention import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()

        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.dropout = dropout


        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)



    def forward(self, x, mask=None):
        B, T, _ = x.size()
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k)

        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        v = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)

        out = scaled_dot_product_attention(
            q, k, v,
            mask=mask,
            dropout_p=self.dropout
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.W_o(out)
