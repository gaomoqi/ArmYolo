import joblib
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from JacobianModel import model



# 创建输出目录
base_output_dir = "training_new_output"
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
output_dir = os.path.join(base_output_dir, f"training_output_new_{new_num}")
os.makedirs(output_dir)

def log_info(file_path, info):
    with open(file_path, 'a') as f:
        f.write(info + '\n')

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path).values
        
        # 分离输入和输出
        self.positions = self.data[:, :6]  # 6列xy位置
        self.tcp_velocities = self.data[:, 6:9]  # 3维TCP速度
        self.outputs = self.data[:, 9:]  # 4维识别框变化速度

        # # 为位置创建StandardScaler
        self.position_scaler = MinMaxScaler(feature_range=(-1,1)).fit(self.positions)
        # # 对位置进行归一化
        self.positions = self.position_scaler.transform(self.positions)
        # # 保存归一化器
        joblib.dump(self.position_scaler, os.path.join(output_dir, 'position_scaler.pkl'))
 
        
    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        # 合并标准化后的位置和TCP速度数据
        inputs = np.concatenate((self.positions[idx], self.tcp_velocities[idx]), axis=0)
        return torch.FloatTensor(inputs), torch.FloatTensor(self.outputs[idx])

# 训练循环
num_epochs = 700
train_losses = []
val_losses = []

# 加载和准备数据
dataset_path = 'cleaned_data/new_cleaned_data_9.csv'
dataset = CustomDataset(dataset_path)
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
criterion = nn.L1Loss()
# criterion = nn.HuberLoss(delta=10.0)
# criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
# 训练函数
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # 分离xy位置和TCP速度
        points, tcp_velocity = inputs[:, :6], inputs[:, 6:]
        
        optimizer.zero_grad()
        outputs = model(points, tcp_velocity)
        prediction = outputs[0]
        jacobian = outputs[1]
        loss = criterion(prediction, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 验证函数
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            # 分离xy位置和TCP速度
            points, tcp_velocity = inputs[:, :6], inputs[:, 6:]

            outputs = model(points, tcp_velocity)
            prediction = outputs[0]
            jacobian = outputs[1]
            loss = criterion(prediction, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


#----------------------------------------------
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_file = os.path.join(output_dir, f"training_log.txt")

# 记录初始信息
log_info(log_file, f"Model structure: {model}")
log_info(log_file, f"Optimizer: {optimizer}")
log_info(log_file, f"Learning rate: {optimizer.param_groups[0]['lr']}")
log_info(log_file, f"Batch size: {train_loader.batch_size}")
log_info(log_file, f"Number of epochs: {num_epochs}")
log_info(log_file, f"dataset: {dataset_path}")
log_info(log_file, f"Train dataset size: {len(train_data)}")
log_info(log_file, f"Validation dataset size: {len(val_data)}")
log_info(log_file, "Epoch,Train Loss,Validation Loss,Learning Rate")
best_val_loss = float('inf')
#----------------------------------------------
# 记录模型参数变化
param_history = {name: [] for name, param in model.named_parameters() if param.requires_grad}

def log_model_params(model, param_history):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_history[name].append(param.data.cpu().numpy().copy())

# 训练循环
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss = validate(model, val_loader, criterion)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
    
    scheduler.step(train_loss)
    
    # 记录模型参数
    log_model_params(model, param_history)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
plt.close()

# 保存最终模型
# torch.save(model, os.path.join(output_dir, 'final_model.pt'))
torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
print(f'Model saved at {output_dir}')
# 记录训练结束信息
log_info(log_file, f"\nTraining completed at: {time.strftime('%Y%m%d-%H%M%S')}")
log_info(log_file, f"Best validation loss: {best_val_loss:.4f}")
log_info(log_file, f"Final training loss: {train_losses[-1]:.4f}")
log_info(log_file, f"Final validation loss: {val_losses[-1]:.4f}")

