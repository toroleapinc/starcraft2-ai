"""CNN policy for spatial actions."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialPolicyNet(nn.Module):
    def __init__(self, in_channels=5, spatial_size=64, n_actions=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        )
        self.spatial_head = nn.Conv2d(32, 1, 1)
        self.fc = nn.Sequential(nn.Linear(32 * spatial_size * spatial_size, 256), nn.ReLU())
        self.action_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        conv_out = self.conv(x)
        spatial = self.spatial_head(conv_out)
        spatial_probs = F.softmax(spatial.view(spatial.size(0), -1), dim=1).view(spatial.size())
        flat = conv_out.view(conv_out.size(0), -1)
        fc_out = self.fc(flat)
        return F.softmax(self.action_head(fc_out), dim=1), spatial_probs, self.value_head(fc_out)
