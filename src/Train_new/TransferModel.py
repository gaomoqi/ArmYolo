import torch
import torch.nn as nn
import torch.optim as optim

from JacobianModel import ImageJacobianNet

class FineTuneModel(ImageJacobianNet):
    def __init__(self):
        super(FineTuneModel, self).__init__()

        for i in range(len(self.layers) - 1):
            for param in self.layers[i].parameters():
                param.requires_grad = False  # 冻结所有层的Linear

        for i in range(len(self.bn_layers)):
            for param in self.bn_layers[i].parameters():
                param.requires_grad = False  # 冻结所有层的BatchNorm1d


    def forward(self, p, xdot):
        return super(FineTuneModel, self).forward(p, xdot)

# 加载预训练模型的权重
pretrained_model_path = 'training_new_output/training_output_new_13/final_model.pt'
pretrained_state_dict = torch.load(pretrained_model_path)

# 创建新模型实例并移动到GPU
fine_tune_model = FineTuneModel().cuda()
# 加载过滤后的状态字典到新模型
fine_tune_model.load_state_dict(pretrained_state_dict, strict=False)

# 打印模型结构，确认哪些层是可训练的
for name, param in fine_tune_model.named_parameters():
    print(f"{name}: {param.requires_grad}")