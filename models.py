import torch
import torch.nn as nn
import torch.nn.functional as F

class FMRI2FACEModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024):
        super(FMRI2FACEModel, self).__init__()
        self.fc = nn.Sequential(
                                nn.Linear(input_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, output_dim),
                                # nn.Linear(input_dim, output_dim),
                                )

    def forward(self, x):
        return self.fc(x)

