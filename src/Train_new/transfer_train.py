import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
import time
from datetime import datetime
from TransferModel import FineTuneModel, fine_tune_model, model_path
import matplotlib.pyplot as plt

base_output_dir = '/home/gaomoqi/ArmYolo_ws/transfer_training_output'
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

# 找到最新的序号
existing_dirs = [d for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d))]
if existing_dirs:
    latest_num = max([int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()])
    new_num = latest_num + 1
else:
    new_num = 1

# 创建新的输出目录
output_dir = os.path.join(base_output_dir, f"transfer_output_{new_num}")
os.makedirs(output_dir)

# 加载数据集
class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path).values
        self.positions = self.data[:, :6]
        self.tcp_velocities = self.data[:, 6:9]
        self.outputs = self.data[:, 9:]

        self.position_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.positions)
        self.positions = self.position_scaler.transform(self.positions)
        joblib.dump(self.position_scaler, os.path.join(output_dir, 'position_scaler.pkl'))

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        inputs = np.concatenate((self.positions[idx], self.tcp_velocities[idx]), axis=0)
        return torch.FloatTensor(inputs), torch.FloatTensor(self.outputs[idx])


dataset_path = 'cleaned_data/new_cleaned_data_10.csv'
dataset = CustomDataset(dataset_path)
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

# 设置损失函数和优化器
criterion = nn.L1Loss()
optimizer = optim.Adam(fine_tune_model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=20, verbose=True)

# 训练和验证函数
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        points, tcp_velocity = inputs[:, :6], inputs[:, 6:]
        optimizer.zero_grad()
        outputs = model(points, tcp_velocity)
        prediction = outputs[0]
        loss = criterion(prediction, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            points, tcp_velocity = inputs[:, :6], inputs[:, 6:]
            outputs = model(points, tcp_velocity)
            prediction = outputs[0]
            loss = criterion(prediction, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


# 训练循环
num_epochs = 200
train_losses = []
val_losses = []
# 记录训练信息
log_file = os.path.join(output_dir, 'training_log.txt')
with open(log_file, 'w') as f:
    f.write(f'Model: {fine_tune_model}\n')
    f.write(f'Base Model path: {model_path}\n')
    f.write(f'Dataset path: {dataset_path}\n')
    f.write(f'Number of epochs: {num_epochs}\n')
    f.write(f'Batch size: {train_loader.batch_size}\n')
    f.write(f'Learning rate: {optimizer.param_groups[0]["lr"]}\n')
    f.write(f'Loss function: {criterion}\n')
    f.write(f'Optimizer: {optimizer}\n')
    f.write(f'Scheduler: {scheduler}\n')

for epoch in range(num_epochs):
    train_loss = train_epoch(fine_tune_model, train_loader, criterion, optimizer)
    val_loss = validate(fine_tune_model, val_loader, criterion)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
    
    scheduler.step(train_loss)

# 保存微调后的模型
torch.save(fine_tune_model, os.path.join(output_dir, 'fine_tuned_model.pt'))

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
plt.close()
