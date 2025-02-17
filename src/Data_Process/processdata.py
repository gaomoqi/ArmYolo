import os
import pandas as pd

def get_next_filename(directory, base_filename):
    """生成递增编号的文件名，确保序号比上一次更大"""
    os.makedirs(directory, exist_ok=True)
    existing_files = [f for f in os.listdir(directory) if f.startswith(base_filename) and f.endswith('.csv')]
    indices = [int(f[len(base_filename)+1:-4]) for f in existing_files if f[len(base_filename)+1:-4].isdigit()]
    next_index = max(indices, default=0) + 1
    return os.path.join(directory, f"{base_filename}_{next_index}.csv")

def process_data(image_file, robot_file, output_dir="cleaned_data"):
    # -------------------------------
    # Step 1: 读取数据
    # -------------------------------
    image_df = pd.read_csv(image_file)
    robot_df = pd.read_csv(robot_file)

    # 解析 robot 数据
    robot_df = robot_df.iloc[:, 0].str.split(expand=True)
    robot_df.columns = [f'TCP_speed_{i}' for i in range(6)]
    robot_df = robot_df.astype(float)

    # 提取所需的图像数据列
    image_selected = image_df[['xc', 'yc', 'width_mid_x', 'width_mid_y',
                               'height_mid_x', 'height_mid_y', 'sample_time']].copy()
    
    # 计算速度 (delta_x / sample_time)
    for col in ['xc', 'yc', 'width_mid_x', 'width_mid_y', 'height_mid_x', 'height_mid_y']:
        image_selected[f'{col}_velocity'] = image_selected[col].diff() / image_selected['sample_time']
    
    # 删除含 NaN 的首行
    image_selected = image_selected.dropna().reset_index(drop=True)
    
    # 删除原始坐标列
    image_selected = image_selected.drop(['xc', 'yc', 'width_mid_x', 'width_mid_y',
                                          'height_mid_x', 'height_mid_y', 'sample_time'], axis=1)
    
    # 对齐 image 数据和 robot 数据
    aligned_data = pd.concat([image_selected, robot_df.iloc[1:].reset_index(drop=True)], axis=1)
    
    # 删除缺失值
    aligned_data = aligned_data.dropna()
    
    # 删除所有列都为零的行
    aligned_data = aligned_data[(aligned_data != 0).any(axis=1)].reset_index(drop=True)
    
    # -------------------------------
    # Step 2: 进一步清理数据
    # -------------------------------
    data_cleaned = aligned_data.dropna()
    
    # 剔除异常值 (基于3倍标准差)
    for col in data_cleaned.columns:
        if 'velocity' in col or 'TCP_speed' in col:
            mean, std = data_cleaned[col].mean(), data_cleaned[col].std()
            data_cleaned = data_cleaned[(data_cleaned[col] >= mean - 3 * std) & 
                                        (data_cleaned[col] <= mean + 3 * std)]
    
    # 剔除无效数据
    velocity_cols = [col for col in data_cleaned.columns if 'velocity' in col]
    robot_cols = [col for col in data_cleaned.columns if 'TCP_speed' in col]
    
    robot_threshold = 1e-3  # TCP 速度接近零的阈值
    image_velocity_threshold = 0.1  # 图像速度异常阈值
    
    mask = (data_cleaned[robot_cols].abs().sum(axis=1) < robot_threshold) & (data_cleaned[velocity_cols].abs().sum(axis=1) > image_velocity_threshold)
    
    data_cleaned = data_cleaned[~mask]
    
    # -------------------------------
    # Step 3: 保存数据
    # -------------------------------
    output_file = get_next_filename(output_dir, "cleaned_data")
    data_cleaned.to_csv(output_file, index=False)
    print(f"数据处理完成，结果已保存为 '{output_file}'")

if __name__ == "__main__":
    image_file = './data_2025_1_20/image_data_2025_01_20_14_24.csv'
    robot_file = './data_2025_1_20/robot_data_2025_01_20_14_24.csv'
    process_data(image_file, robot_file)
