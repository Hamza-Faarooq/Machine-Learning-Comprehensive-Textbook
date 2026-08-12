import torch
import numpy as np

class DDPMForwardProcess:
    """
    Implements the forward (noising) process for Diffusion Models[cite: 1].
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        self.num_timesteps = num_timesteps
        
        # Linear variance schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Adds noise to the original image x_0 to reach timestep t.
        Equation: q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)I)
        """
        noise = torch.randn_like(x_0)
        
        # Extract the precomputed values for the requested timesteps
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])
        
        # Reshape for broadcasting over the image dimensions
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        
        # Compute the noisy image x_t
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Dummy batch of 1 RGB image, 32x32
    x_0_dummy = torch.randn(1, 3, 32, 32)
    
    # Sample a random timestep
    t_dummy = torch.tensor([500])
    
    ddpm = DDPMForwardProcess(num_timesteps=1000)
    x_t_dummy, added_noise = ddpm.add_noise(x_0_dummy, t_dummy)
    
    print("--- DDPM Forward Process ---")
    print(f"Original Image Mean: {x_0_dummy.mean().item():.1f}")
    print(f"Noisy Image Mean:    {x_t_dummy.mean().item():.1f}")
