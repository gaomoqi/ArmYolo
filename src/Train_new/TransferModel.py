import torch
import torch.nn as nn
import torch.optim as optim
# 加载之前训练好的模型
model_path = 'training_new/training_output_new_with_norm/final_model.pt'
model = torch.load(model_path,weights_only=False)

# 冻结之前的参数
for param in model.parameters():
    param.requires_grad = False

# 添加新的层
class FineTuneModel(nn.Module):
    def __init__(self, base_model ,hidden_sizes=[64, 64], jacobian_size=12, dropout_rate=0):
        super(FineTuneModel, self).__init__()
        self.base_model = base_model
        self.new_layer_list = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()

        for size in hidden_sizes:
            self.batch_norm_list.append(nn.BatchNorm1d(size))

        self.new_layer_list.append(nn.Linear(12, hidden_sizes[0]))#12 jacobian_size input
        for i in range(len(hidden_sizes) - 1):
            self.new_layer_list.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.new_layer_list.append(nn.Linear(hidden_sizes[-1], jacobian_size))

        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, points, tcp_velocity):
        prediction, jacobian = self.base_model(points, tcp_velocity)
        x = jacobian.view(-1, 12)
        
        for i,layer in enumerate(self.new_layer_list[:-1]):
            x = layer(x)
            x = self.batch_norm_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.new_layer_list[-1](x)#12 jacobian_size output
        new_jacobian = x.view(-1, 4, 3)
        output = torch.bmm(new_jacobian, tcp_velocity.unsqueeze(2)).squeeze(2)
        return output, new_jacobian

# 初始化新的模型
fine_tune_model = FineTuneModel(model).cuda()