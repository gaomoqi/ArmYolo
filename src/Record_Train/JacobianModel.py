from torch import nn
class ImageJacobianNet(nn.Module):
    def __init__(self, input_size=6, hidden_sizes=[64, 128, 64], output_size=3, dropout_rate=0.5):
        super(ImageJacobianNet, self).__init__()
        self.layers = nn.ModuleList()
        
        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # 添加多个隐藏层
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        # 最后一个隐藏层到输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        # 最后一层不应用LayerNorm、ReLU和Dropout
        x = self.layers[-1](x)
        return x
    
# 创建模型实例
model = ImageJacobianNet(input_size=4, 
                         hidden_sizes=[64, 128, 256, 128, 64], 
                         output_size=3, 
                         dropout_rate=0.2).cuda()