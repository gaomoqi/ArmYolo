import numpy as np
from torch import nn
import torch

# class ImageJacobianNet(nn.Module):
#     def __init__(self, input_size=6, hidden_sizes=[64], jacobian_size=12, dropout_rate=0.5):
#         super(ImageJacobianNet, self).__init__()
#         self.layers = nn.ModuleList()
#         self.norms = nn.ModuleList()

#         # 输入层到第一个隐藏层
#         self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
#         # 添加多个隐藏层
#         for i in range(len(hidden_sizes) - 1):
#             self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
#         # 最后一个隐藏层到输出层
#         self.layers.append(nn.Linear(hidden_sizes[-1], jacobian_size))

#         # 添加BatchNorm层
#         for size in hidden_sizes:
#             self.norms.append(nn.BatchNorm1d(size))

#         self.relu = nn.LeakyReLU(negative_slope=0.01)
#         self.dropout = nn.Dropout(dropout_rate)
        
#     def forward(self, points, tcp_velocity):
        
#         x = points
#         for i, layer in enumerate(self.layers[:-1]):
#             x = layer(x)
#             x = self.norms[i](x)
#             x = self.relu(x)
#             x = self.dropout(x)

#         # 最后一层不应用ReLU和Dropout
#         x = self.layers[-1](x)
        
#         # 将输出重塑为4x3的雅可比矩阵
#         jacobian = x.view(-1, 4, 3)
        
#         # 将雅可比矩阵与TCP速度相乘
#         output = torch.bmm(jacobian, tcp_velocity.unsqueeze(2)).squeeze(2)
        
#         return output, jacobian

# # 创建模型实例
# model = ImageJacobianNet(input_size=6, 
#                          hidden_sizes=[64,128,64], 
#                          jacobian_size=12, 
#                          dropout_rate=0).cuda()


class ImageJacobianNet(nn.Module):
    def __init__(self):
        super(ImageJacobianNet, self).__init__()
        self.fc1 = nn.Linear(6, 60, bias=True)
        # self.fc2 = nn.Linear(64, 128, bias=True)
        # self.fc3 = nn.Linear(128, 60, bias=True)
        self.fc4 = nn.Linear(60, 4, bias=False)
        self.leaky_relu1 = nn.LeakyReLU(0.01)

    def forward(self, p, xdot):
        # p: shape (N, 6)
        # xdot: shape (N, 6)
        fc1_output = self.leaky_relu1(self.fc1(p))
        part1, part2, part3 = torch.split(fc1_output, 20, dim=1)

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
        return J_final.view(-1, 4),jacobian

model = ImageJacobianNet().cuda()