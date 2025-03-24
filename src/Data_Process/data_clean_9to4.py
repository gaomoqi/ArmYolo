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
    robot_df = pd.read_csv(robot_file,sep=r'\s+')
    # 选择需要的列
    # image_selected = image_df[['xc', 'yc', 'width_mid_x', 'width_mid_y',
    #                            'height_mid_x', 'height_mid_y', 'sample_time']].copy()
    image_selected = image_df[['xc', 'yc', 'x1_x','x1_y','x2_x','x2_y','x3_x','x3_y','x4_x','x4_y','sample_time']].copy()
    robot_selected = robot_df[['actual_TCP_speed_0', 'actual_TCP_speed_1', 'actual_TCP_speed_2']].copy()

    # 计算宽度和高度
    image_selected['width'] = ((image_selected['x2_x'] - image_selected['x1_x'])**2 + 
                               (image_selected['x2_y'] - image_selected['x1_y'])**2)**0.5
    image_selected['height'] = ((image_selected['x4_x'] - image_selected['x1_x'])**2 + 
                                (image_selected['x4_y'] - image_selected['x1_y'])**2)**0.5

    # 计算速度 (delta_x / sample_time)
    for col in ['xc', 'yc', 'width', 'height']:
        image_selected[f'{col}_velocity'] = image_selected[col].diff() / image_selected['sample_time']
    

    # 删除含 NaN 的首行
    image_selected = image_selected.dropna().reset_index(drop=True)
    robot_selected = robot_selected.dropna().reset_index(drop=True)
    # 删除原始坐标列
    image_selected = image_selected.drop(['xc','yc','x1_x','x1_y','x2_x','x2_y','x3_x','x3_y','x4_x','x4_y','width', 'height','sample_time'], axis=1)
    
    position_selected = image_df[['xc','yc','width_mid_x', 'width_mid_y','height_mid_x', 'height_mid_y']].copy()

    # 对齐 image 数据和 robot 数据
    min_rows = min(len(image_selected), len(robot_selected))
    image_selected = image_selected.iloc[:min_rows]
    robot_selected = robot_selected.iloc[:min_rows]

    # 横向合并两个数据集
    aligned_data = pd.concat([position_selected, robot_selected, image_selected], axis=1)

    # 打印结果确认
    # print(aligned_data.columns)
    
    # 删除缺失值
    data_cleaned = aligned_data.dropna()

    # 删除图像速度为0的行和TCP速度为0的行
    image_velocity_cols = [col for col in data_cleaned.columns if 'velocity' in col]
    tcp_speed_cols = [col for col in data_cleaned.columns if 'TCP_speed' in col]
    epsilon = 1e-6  # 定义一个小的阈值
    data_cleaned = data_cleaned[~(abs(data_cleaned[image_velocity_cols]) < epsilon).all(axis=1)]
    data_cleaned = data_cleaned[~(abs(data_cleaned[tcp_speed_cols]) < epsilon).all(axis=1)]
    
    # 重置索引
    data_cleaned = data_cleaned.reset_index(drop=True)

     # 剔除异常值 (基于3倍标准差)
    for col in data_cleaned.columns:
        if 'velocity' in col or 'TCP_speed' in col:
            mean, std = data_cleaned[col].mean(), data_cleaned[col].std()
            data_cleaned = data_cleaned[(data_cleaned[col] >= mean - 3 * std) & 
                                        (data_cleaned[col] <= mean + 3 * std)]
    
    # 剔除速度突变值，分别处理图像速度和TCP速度
    def remove_acceleration_outliers(df, cols, std_multiplier=5):
        for col in cols:
            acceleration = df[col].diff()
            acc_mean, acc_std = acceleration.mean(), acceleration.std()
            threshold = std_multiplier * acc_std
            df = df[abs(acceleration - acc_mean) <= threshold]
        return df

    # 处理图像速度的加速度异常
    # data_cleaned = remove_acceleration_outliers(data_cleaned, image_velocity_cols)
    
    # 处理TCP速度的加速度异常
    # data_cleaned = remove_acceleration_outliers(data_cleaned, tcp_speed_cols)
    
    # 重置索引
    data_cleaned = data_cleaned.reset_index(drop=True)
    # -------------------------------
    # Step 3: 保存数据
    # -------------------------------
    output_file = get_next_filename(output_dir, "new_cleaned_data")
    data_cleaned.to_csv(output_file, index=False)
    print(f"数据处理完成，结果已保存为 '{output_file}'")

if __name__ == "__main__":
    image_file = 'original_data/data_2025_3_20/image_data_2025_03_20_21_07.csv'
    robot_file = 'original_data/data_2025_3_20/robot_data_2025_03_20_21_07.csv'
    process_data(image_file, robot_file)
