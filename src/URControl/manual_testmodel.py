import torch
import torch.nn as nn
import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import time
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Train_new")))
# from JacobianModel_new import ImageJacobianNet
from TransferModel import FineTuneModel
"""
xc,yc,width_mid_x,width_mid_y,height_mid_x,height_mid_y,actual_TCP_speed_0,actual_TCP_speed_1,actual_TCP_speed_2,xc_velocity,yc_velocity,width_velocity,height_velocity
320,123,325,233,239,127,-0.044697246447419,-0.120019833337247,-0.061317288754061,-84.93851820438661,39.97106739029958,80.88012628875734,97.769117932547

"""

"""Transfer
xc,yc,width_mid_x,width_mid_y,height_mid_x,height_mid_y,actual_TCP_speed_0,actual_TCP_speed_1,actual_TCP_speed_2,xc_velocity,yc_velocity,width_velocity,height_velocity
355,310,355.5,412,268,310.5,-0.1065791202609424,-0.0577238032622549,-0.0002178817382746,225.68059775376045,-120.36298546867225,-15.045373183584031,-50.529834942494354
"""

# position_scaler = joblib.load("training_new_output/training_output_new_1/position_scaler.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model_test = torch.load("transfer_training_output/transfer_output_8/fine_tuned_model.pt",weights_only=False)
position_scaler = joblib.load("transfer_training_output/transfer_output_8/position_scaler.pkl")
# 切换到评估模式
model_test.eval()

position_origin = np.array([355,310,355,412,268,310])
position_test = torch.tensor(position_scaler.transform(position_origin.reshape(1, -1)), dtype=torch.float32).to(device)
# xdot = torch.tensor(np.array([-0.044697246447419,-0.120019833337247,-0.061317288754061]).reshape(1, -1), dtype=torch.float32).to(device)
xdot  = torch.tensor(np.array([1,1,1]).reshape(1,-1),dtype=torch.float32).to(device)
image_speed_cmd = np.array([225.68059775376045,-120.36298546867225,-15.045373183584031,-50.529834942494354])
with torch.no_grad():
    output_test = model_test(position_test, xdot)
    print("output_test:",output_test[0],"\n",output_test[1])
    jacobian_matrix = output_test[1].to('cpu')#据位置信息预测雅可比矩阵
    jacobian_pinv = np.linalg.pinv(jacobian_matrix)  # 计算 (3, 4) 形状的伪逆
    v = np.dot(jacobian_pinv, image_speed_cmd).flatten()  # 计算 (3, 4) @ (4,) -> (3,)
    print("v_cmd:",v)

    
    