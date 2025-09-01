import torch 
import torch.nn.functional as f
import math


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)


    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    

    attn = f.softmax(scores, dim=-1)
    attn = f.dropout(attn, p=dropout_p)
    out = torch.matmul(attn, v)
    return out