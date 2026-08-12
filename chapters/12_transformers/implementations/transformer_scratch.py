import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Scaled Dot-Product Multi-Head Attention from scratch[cite: 1].
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0.0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        batch_size, seq_len, _ = Q.shape
        
        # Linear projections and reshape for multi-head: (Batch, Heads, SeqLen, d_k)
        q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Mask out invalid positions (e.g., padding or future tokens)
            scores = scores.masked_fill(mask == 0.0, float('-inf'))
            
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        
        # Aggregate values
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights

class TransformerEncoderLayer(nn.Module):
    """
    A single Transformer Encoder block utilizing Pre-Norm architecture[cite: 1].
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-Norm architecture: Normalize before attention/FFN
        nx = self.norm1(x)
        attn_out, _ = self.self_attn(nx, nx, nx, mask)
        x = x + self.dropout(attn_out)
        
        nx2 = self.norm2(x)
        ffn_out = self.ffn(nx2)
        x = x + self.dropout(ffn_out)
        
        return x

def test_transformer() -> None:
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 16
    d_model = 256
    n_heads = 8
    
    # Dummy input sequence (e.g., embedded tokens)
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    
    encoder_layer = TransformerEncoderLayer(d_model=d_model, n_heads=n_heads)
    output = encoder_layer(dummy_input)
    
    print("--- Transformer Encoder Layer ---")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")

if __name__ == "__main__":
    test_transformer()
