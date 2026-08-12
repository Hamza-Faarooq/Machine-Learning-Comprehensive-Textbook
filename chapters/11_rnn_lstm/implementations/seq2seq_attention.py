import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BahdanauAttention(nn.Module):
    """
    Additive attention mechanism[cite: 1].
    """
    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.W_s = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.W_h = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, s_prev: torch.Tensor, encoder_outputs: torch.Tensor) -> tuple:
        # s_prev: (Batch, dec_dim), encoder_outputs: (Batch, Seq_len, enc_dim)
        
        # Calculate energy scores
        energy = self.v(torch.tanh(
            self.W_s(s_prev).unsqueeze(1) + self.W_h(encoder_outputs)
        )).squeeze(2) # (Batch, Seq_len)
        
        # Normalize to attention weights
        alpha = F.softmax(energy, dim=1)
        
        # Compute context vector
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, alpha

class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder for sequence compression[cite: 1].
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layers: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, n_layers, 
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        
        # Project bidirectional hidden states to unidirectional size for decoder
        self.proj_h = nn.Linear(2 * hidden_dim, hidden_dim)
        self.proj_c = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> tuple:
        embeds = self.dropout(self.embedding(src))
        
        # Pack sequence to ignore padding tokens during LSTM computation
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        outputs, (h_n, c_n) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Concatenate forward and backward final states
        h_n_concat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        c_n_concat = torch.cat([c_n[-2], c_n[-1]], dim=1)
        
        # Project to match decoder dimensions
        h_n_proj = torch.tanh(self.proj_h(h_n_concat)).unsqueeze(0)
        c_n_proj = torch.tanh(self.proj_c(c_n_concat)).unsqueeze(0)
        
        return outputs, (h_n_proj, c_n_proj)

def test_encoder() -> None:
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 10
    vocab_size = 1000
    
    # Dummy input sequences and lengths
    dummy_src = torch.randint(1, vocab_size, (batch_size, seq_len))
    dummy_lengths = torch.tensor([10.0, 8.0, 6.0, 5.0])
    
    encoder = BiLSTMEncoder(vocab_size=vocab_size, embed_dim=64, hidden_dim=128)
    outputs, (h_n, c_n) = encoder(dummy_src, dummy_lengths)
    
    print("--- BiLSTM Encoder Shape Check ---")
    print(f"Encoder Outputs Shape: {outputs.shape}") # Expected: (4, 10, 256)
    print(f"Hidden State Shape:    {h_n.shape}")     # Expected: (1, 4, 128)
    
if __name__ == "__main__":
    test_encoder()
