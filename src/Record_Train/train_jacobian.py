import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 创建输出文件夹
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
output_folder = f'training_output{timestamp}'
os.makedirs(output_folder, exist_ok=True)

class JacobianDataset(Dataset):
    def __init__(self, data_file):
        # 加载你的数据文件
        self.data = pd.read_csv(data_file).values
        self.inputs = self.data[:, :6]  
        self.targets = self.data[:, 6:]  

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = torch.FloatTensor(self.inputs[idx])
        target = torch.FloatTensor(self.targets[idx])
        return input_data, target
# dataset = JacobianDataset('cleaned_data_2.csv')
# 设置训练集和验证集的比例 随机分割数据集
# train_size = int(0.8 * len(dataset))  # 80% 用于训练
# val_size = len(dataset) - train_size  # 剩下的 20% 用于验证

# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset = JacobianDataset('cleaned_data_2.csv')
val_dataset = JacobianDataset('cleaned_data.csv')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#-----------------两层神经网络--------------------#
# class ImageJacobianNet(nn.Module):
#     def __init__(self, input_size=6, hidden_size=64, output_size=6):
#         super(ImageJacobianNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
# model = ImageJacobianNet().cuda()  # 移到GPU

#------------------多层神经网络--------------------#
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

# 创建模型实例
model = ImageJacobianNet(input_size=6, 
                         hidden_sizes=[64, 128, 256, 128, 64], 
                         output_size=6, 
                         dropout_rate=0.5).cuda()
print(model)

# 定义损失函数
# criterion = nn.MSELoss()

# 定义带有L1正则化的损失函数
class L1RegularizedMSELoss(nn.Module):
    def __init__(self, l1_lambda=0.01):
        super(L1RegularizedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_lambda = l1_lambda

    def forward(self, predictions, targets, model):
        mse = self.mse_loss(predictions, targets)
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        return mse + self.l1_lambda * l1_reg
criterion = L1RegularizedMSELoss(l1_lambda=0.01)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 学习率衰减策略
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# 训练模型
num_epochs = 100
losses = []
val_losses = []  
param_history = {name: [] for name, _ in model.named_parameters()}

for epoch in range(num_epochs):
    model.train()  
    epoch_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets, model)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    for name, param in model.named_parameters():
        param_history[name].append((param.data.mean().item(), param.data.std().item()))

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}')

    # 计算验证集上的损失
    model.eval()  
    epoch_val_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            val_loss = criterion(outputs, targets, model)
            epoch_val_loss += val_loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)  
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
    
    # 更新学习率
    scheduler.step(avg_val_loss)

# 保存模型
torch.save(model, 'jacobian_model.pt')

# 绘制训练损失和验证损失
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_folder, 'training_validation_loss.png'))
plt.show()

# 绘制模型参数变化
for name, history in param_history.items():
    means, stds = zip(*history)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), means, label='Mean')
    plt.plot(range(1, num_epochs+1), stds, label='Std')
    plt.title(f'Parameter {name} Changes During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'{name}_changes.png'))
    plt.close()
    plt.show()

# # 可视化模型参数
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         plt.figure(figsize=(10, 5))
#         plt.hist(param.data.cpu().numpy().flatten(), bins=50)
#         plt.title(f'Histogram of {name}')
#         plt.xlabel('Value')
#         plt.ylabel('Frequency')
#         plt.savefig(f'{name}_histogram.png')
#         plt.show()
