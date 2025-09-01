import torch

def casual_mask(seq_len : int, device="cpu"):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

    return mask