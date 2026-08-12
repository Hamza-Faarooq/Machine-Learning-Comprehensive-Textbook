import torch
import torch.nn as nn
import torch.optim as optim
from typing import Type

class ResidualBlock(nn.Module):
    """
    Standard ResNet Residual Block with skip connections.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection for matching dimensions when stride > 1
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out += identity
        out = self.relu(out)
        return out

class ResNetSmall(nn.Module):
    """
    A lightweight ResNet variant for image classification tasks.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 32
        
        # Initial stem
        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Residual Stages
        self.layer1 = ResidualBlock(32, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        
        # Global Average Pooling & Classifier Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prep(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def run_cnn_training_pipeline() -> None:
    """Executes a single training epoch of the custom ResNet on synthetic images."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Batch of 8 RGB images of size 32x32
    batch_size = 8
    dummy_images = torch.randn(batch_size, 3, 32, 32).to(device)
    dummy_labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    model = ResNetSmall(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Forward Pass & Optimization
    model.train()
    optimizer.zero_grad()
    
    outputs = model(dummy_images)
    loss = criterion(outputs, dummy_labels)
    
    loss.backward()
    optimizer.step()
    
    # Calculate Training Accuracy
    preds = torch.argmax(outputs, dim=1)
    acc = (preds == dummy_labels).float().mean() * 100.0
    
    print("--- PyTorch ResNet Pipeline ---")
    print(f"Batch Training Loss:     {loss.item():.1f}")
    print(f"Batch Training Accuracy: {acc.item():.1f}%")

if __name__ == "__main__":
    run_cnn_training_pipeline()
