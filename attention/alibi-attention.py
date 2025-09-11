import torch
import torch.nn as nn
import math


class ALiBiAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim, dim)
        self.max_seq_len = max_seq_len
        self.register_buffer('alibi', self._build_alibi_bias(max_seq_len, num_heads))

    def _build_alibi_bias(self, max_seq_len, num_heads):
        slopes = torch.logspace(-4, -0.3, steps=num_heads, base=2.0)  
        pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(0)  
        bias = pos.unsqueeze(1) * slopes.unsqueeze(0).unsqueeze(-1) 
        bias = -bias.unsqueeze(2) 
        return bias

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        alibi_bias = self.alibi[:, :, :, :L]
        scores = scores + alibi_bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.out(out)
    


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.attn = ALiBiAttention(dim, num_heads, max_seq_len, dropout)
        self.ff = FeedForward(dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))


        return x


class ALiBiTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, hidden_dim, max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, hidden_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(self, x, mask=None):
        B, L = x.shape

        x = self.embedding(x) + self.pos_encoding[:,  :L, :]

        x = self.dropout(x)
        for layer in self.layers :
            x = layer(x, mask)
        
        x = self.norm(x)
        return self.head(x)
    

def create_casual_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(1)




def train_model():
    vocab_size = 10000
    dim = 256
    num_heads = 8
    num_layers = 4
    hidden_dim = 512
    max_seq_len = 512
    batch_size = 32
    seq_len = 128
    lr = 3e-4
    epochs = 10


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ALiBiTransformer(vocab_size, dim, num_heads, num_layers, hidden_dim, max_seq_len,).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()


    def generate_dummy_data(batch_size, seq_len, vocab_size):
        return torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    model.train()


    for epoch in range(epochs):
        total_loss = 0
        for _ in range(100):
            inputs = generate_dummy_data(batch_size, seq_len, vocab_size)
            targets = inputs[:, 1:].contiguous()
            mask = create_casual_mask(seq_len).to(device)

            optimizer.zero_grad()
            outputs = model(inputs, mask)[:, :-1, :].contiguous()
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))


            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        print(f'Epoch {epoch + 1}, Loss : {total_loss / 100:.4f}')


    
    model.eval()
    test_seq_len = 1024
    test_input = generate_dummy_data(1, test_seq_len, vocab_size)
    test_mask = create_casual_mask(test_seq_len).to(device)


    with torch.no_grad():
        output = model(test_input, test_mask)

    print(f'test output shape for seq_len {test_seq_len} : {output.shape}')


if __name__ == '__main__':
    train_model()
