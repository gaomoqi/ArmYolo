import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('/home/gaomoqi/ArmYolo_ws/cleaned_data/new_cleaned_data_8.csv')

# 定义一个函数来检测突变
def detect_sudden_changes(series, threshold=2):
    # 计算相邻点之间的差值
    diff = series.diff()
    
    # 计算差值的均值和标准差
    mean_diff = diff.mean()
    std_diff = diff.std()
    
    # 计算Z-score
    z_score = np.abs((diff - mean_diff) / std_diff)
    
    # 将第一个值的Z-score设为0（因为它没有前一个值）
    z_score.iloc[0] = 0
    
    # 返回一个布尔序列，True表示是突变点
    return z_score > threshold

# 获取最后四列的列名（假设是速度数据）
velocity_columns = df.columns[-4:]

# 对每一列速度数据应用突变检测
sudden_change_mask = pd.DataFrame()
for col in velocity_columns:
    sudden_change_mask[col] = detect_sudden_changes(df[col])

# 只保留所有列都不是突变点的行
mask = ~sudden_change_mask.any(axis=1)
df_cleaned = df[mask]

# 将清理后的数据保存到新的CSV文件
output_file = '/home/gaomoqi/ArmYolo_ws/cleaned_data/new_cleaned_data_8_cleaned.csv'
df_cleaned.to_csv(output_file, index=False)

print(f"原始数据行数: {len(df)}")
print(f"清理后数据行数: {len(df_cleaned)}")
print(f"清理后的数据已保存到: {output_file}")