import torch
import torch.nn as nn
import torch.optim as optim

class SimpleAutoencoder(nn.Module):
    """
    A symmetric feedforward autoencoder for dimensionality reduction[cite: 1].
    """
    def __init__(self, input_dim: int = 784, latent_dim: int = 32) -> None:
        super().__init__()
        
        # Encoder: Compresses the input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder: Reconstructs the input from the latent code
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid() # Assumes normalized inputs between [0, 1]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_code = self.encoder(x)
        reconstruction = self.decoder(latent_code)
        return reconstruction

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Simulate a batch of 16 flattened 28x28 images (e.g., MNIST)
    batch_size = 16
    input_dim = 784
    dummy_input = torch.rand((batch_size, input_dim))
    
    model = SimpleAutoencoder(input_dim=input_dim, latent_dim=32)
    criterion = nn.MSELoss() # Reconstruction loss[cite: 1]
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Single training step
    model.train()
    optimizer.zero_grad()
    
    output = model(dummy_input)
    loss = criterion(output, dummy_input)
    
    loss.backward()
    optimizer.step()
    
    print("--- PyTorch Autoencoder ---")
    print(f"Initial Reconstruction Loss: {loss.item():.1f}")
