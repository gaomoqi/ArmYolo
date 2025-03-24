import numpy as np
from torch import nn
import torch

class ImageJacobianNet(nn.Module):
    def __init__(self):
        super(ImageJacobianNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(6, 64, bias=True),
            nn.Linear(64, 128, bias=True),
            nn.Linear(128, 60, bias=True)
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(64),
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(60)
        ])
        self.fc4 = nn.Linear(60, 4, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.01)
    
    def forward(self, p, xdot):
        # p: shape (N, 6)
        # xdot: shape (N, 6)
        x = p
        for layer, bn in zip(self.layers, self.bn_layers):
            x = layer(x)
            x = bn(x)
            x = self.leaky_relu(x)
        
        fc3_output = x
        part1, part2, part3 = torch.split(fc3_output, 20, dim=1)

        # Extract the 3 components of xdot
        xdot1 = xdot[:, 0].view(-1, 1)
        xdot2 = xdot[:, 1].view(-1, 1)
        xdot3 = xdot[:, 2].view(-1, 1)

        # Multiply each part by its corresponding xdot component
        part1 = part1 * xdot1
        part2 = part2 * xdot2
        part3 = part3 * xdot3
        
        # Split the fc4 weights into 3 parts (each of size corresponding to 20 dimensions)
        W1, W2, W3 = torch.split(self.fc4.weight, 20, dim=1)
        J1 = torch.matmul(part1, W1.T)
        J2 = torch.matmul(part2, W2.T)
        J3 = torch.matmul(part3, W3.T)
        J_final = J1 + J2 + J3 

        jacobian = torch.stack([J1, J2, J3], dim=2)  # 形状变为 (batch_size, 4, 3)
        return J_final.view(-1, 4), jacobian

model = ImageJacobianNet().cuda()