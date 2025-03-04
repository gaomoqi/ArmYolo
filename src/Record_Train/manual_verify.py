import torch
import torch.nn as nn
import numpy 
import joblib
import os

input_scaler = joblib.load('training_output2025_02_25_16_25/input_scaler.pkl')
output_scaler = joblib.load('training_output2025_02_25_16_25/output_scaler.pkl')
class ImageJacobianNet(nn.Module):
    def __init__(self, input_size=6, hidden_sizes=[64, 128, 64], output_size=6, dropout_rate=0.5):
        super(ImageJacobianNet, self).__init__()
        self.layers = nn.ModuleList()
        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        # 添加多个隐藏层
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        # 最后一个隐藏层到输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 不在最后一层应用ReLU和Dropout
                x = self.relu(x)
                x = self.dropout(x)
        return x


model = ImageJacobianNet(input_size=6, 
                         hidden_sizes=[64, 128, 256, 128, 64], 
                         output_size=3, 
                         dropout_rate=0.5)

model = torch.load("training_output2025_02_25_16_25/jacobian_model_xyz.pt", weights_only=False)
model.eval()
model.cpu()


with torch.no_grad():
    input_data = numpy.array([100, -18, 99, -3, 98, -18]).flatten()# 假设输入数据是一个6维的张量
    normalized_input = input_scaler.transform(input_data.reshape(1, -1))

    input_tensor = torch.FloatTensor(input_data).unsqueeze(0).cpu()
    output = model(input_tensor).squeeze(0).numpy().flatten()

    output_denormalized = output_scaler.inverse_transform(output.reshape(1,-1)).flatten()
    print(output_denormalized)
