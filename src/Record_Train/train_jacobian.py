import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import random_split
import joblib
from JacobianModel import model

#-----------------------------------------#
initial_lr = 0.0001#学习率
initial_batch_size = 32#批大小
train_dataset_path = "cleaned_data/cleaned_data_6.csv"
print(model)
#------------------------------------------#


# 创建输出文件夹
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
output_folder = f'training_output{timestamp}'
os.makedirs(output_folder, exist_ok=True)
class JacobianDataset(Dataset):
    def __init__(self, data_file):
        # 加载数据
        self.data = pd.read_csv(data_file).values
        self.inputs = self.data[:, :4]  
        self.targets = self.data[:, 4:]

        # 对输入数据进行归一化
        self.input_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.input_scaler.fit(self.inputs) 
        self.inputs = self.input_scaler.transform(self.inputs)  

        self.output_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.output_scaler.fit(self.targets)
        self.outputs = self.output_scaler.transform(self.targets)  

        # 保存归一化的 scaler
        joblib.dump(self.input_scaler, f'{output_folder}/input_scaler.pkl')
        joblib.dump(self.output_scaler, f'{output_folder}/output_scaler.pkl')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = torch.FloatTensor(self.inputs[idx])
        target = torch.FloatTensor(self.outputs[idx])
        return input, target

# 加载数据并进行归一化
train_dataset = JacobianDataset(train_dataset_path)

# 分割训练集和验证集
train_size = int(0.8 * len(train_dataset))  # 80% 用于训练
val_size = len(train_dataset) - train_size  # 20% 用于验证

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# train_dataset = JacobianDataset('cleaned_data/cleaned_data_5(only_xyz).csv')
# val_dataset = JacobianDataset('cleaned_data/cleaned_data_5(only_xyz).csv')
train_loader = DataLoader(train_dataset, batch_size=initial_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=initial_batch_size, shuffle=False)


# ----------------定义损失函数--------------------#
criterion = nn.MSELoss()
# criterion = nn.HuberLoss(delta=0.05) 

# --------------定义验证损失函数------------------#
val_criterion = nn.MSELoss()    
# val_criterion = nn.HuberLoss(delta=0.05) 


# # 定义带有L1正则化的损失函数
# class L1RegularizedMSELoss(nn.Module):
#     def __init__(self, l1_lambda=0.01):
#         super(L1RegularizedMSELoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.l1_lambda = l1_lambda

#     def forward(self, predictions, targets, model):
#         mse = self.mse_loss(predictions, targets)
#         l1_reg = sum(torch.norm(param, 1) for param in model.parameters())
#         return mse + self.l1_lambda * l1_reg
# criterion = L1RegularizedMSELoss(l1_lambda=0.0001)#0.01

# 训练模型
num_epochs = 100
losses = []
val_losses = []  
param_history = {name: [] for name, _ in model.named_parameters()}
batch_losses = []


# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

# 学习率衰减策略
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
gradients = {name: [] for name, _ in model.named_parameters()}  # 记录每个参数的梯度均值

for epoch in range(num_epochs):
    model.train()  
    epoch_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:  # 检查梯度是否存在
                gradients[name].append(param.grad.mean().item())
        optimizer.step()

        batch_losses.append(loss.item())  # 记录 batch loss
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
            val_loss = val_criterion(outputs, targets)
            epoch_val_loss += val_loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)  
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

    scheduler.step()  # 更新学习率

    

# 保存模型
torch.save(model, f'training_output{timestamp}/jacobian_model_xyz.pt')

# 绘制训练损失和验证损失
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_folder, f'training_validation_loss().png'))
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


# for name, grad_values in gradients.items():
#     plt.plot(grad_values, label=name)  # 画出每个参数的梯度变化

#     plt.xlabel("Iterations (batches)")
#     plt.ylabel("Mean Absolute Gradient")
#     plt.title("Gradient Change During Training")
#     plt.legend()
#     plt.grid()
#     plt.savefig(os.path.join(output_folder, 'gradients_plot.png'))


# 记录训练信息到txt文件
with open(os.path.join(output_folder, "training_info.txt"), "w") as f:
    f.write("模型结构:\n")
    f.write(str(model) + "\n\n")
    f.write("\n\n损失函数: MSELoss()\n")
    f.write(f"学习率: {initial_lr}\n")
    f.write(f"学习率衰减策略: CosineAnnealingLR\n")
    f.write(f"Batch大小: {initial_batch_size}\n")
    f.write(f"训练集: {train_dataset_path}\n")  # 请替换为实际的训练集路径变量
    f.write(f"总训练轮数: {num_epochs}\n")
    f.write(f"最终训练损失: {losses[-1]:.4f}\n")
    f.write(f"最终验证损失: {val_losses[-1]:.4f}\n")

print("训练信息已保存到", os.path.join(output_folder, "training_info.txt"))


