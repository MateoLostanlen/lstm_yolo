import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    EfficientNet_B1_Weights,
)  # Import the correct weights enum


class TemporalModel(nn.Module):
    def __init__(self):
        super(TemporalModel, self).__init__()

        # Load a pre-trained EfficientNet-B1 model using weights instead of pretrained
        efficientnet = models.efficientnet_b1(
            weights=EfficientNet_B1_Weights.IMAGENET1K_V1
        )
        self.efficientnet = nn.Sequential(
            *list(efficientnet.children())[:-2]
        )  # Remove the final layers

        # Freeze EfficientNet weights
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Global Average Pooling equivalent
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=1280, hidden_size=256, num_layers=3, batch_first=True
        )

        # Final fully connected layer for binary classification (1 output neuron)
        self.fc = nn.Linear(256, 1)

        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()

        # Reshape input to combine batch and sequence dimensions
        x = x.view(batch_size * seq_len, C, H, W)

        # Pass through EfficientNet backbone
        x = self.efficientnet(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output (batch_size * seq_len, 1280)

        # Reshape back to (batch_size, seq_len, 1280) for LSTM input
        x = x.view(batch_size, seq_len, -1)

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Use only the output of the last timestep for classification
        x = x[:, -1, :]

        # Pass through final fully connected layer and apply sigmoid
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
