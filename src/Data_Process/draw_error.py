import matplotlib.pyplot as plt
import cv2
import numpy as np
import csv

# 读取CSV文件
filename = 'record_error/record_error_2025_03_19_19_10/error_log.csv'
video_path = 'record_error/record_error_2025_03_19_19_10/output.mp4'

data = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # 跳过标题行
    for row in csvreader:
        try:
            # 尝试将所有值转换为浮点数
            data.append([float(value) for value in row])
        except ValueError:
            # 如果转换失败,跳过该行
            continue

# 每隔5个点取一个点
data_sampled = data[::5]

# 提取时间戳和误差数据
timestamps = [row[0] for row in data_sampled]
errors = [row[2:] for row in data_sampled]

# 创建主图和子图
fig, (ax_main, ax_frames) = plt.subplots(2, 1, figsize=(20, 15), height_ratios=[3, 1])

# 绘制误差曲线
ax_main.plot(timestamps, [error[0] for error in errors], label=f'error_x')
ax_main.plot(timestamps, [error[1] for error in errors], label=f'error_y')
ax_main.plot(timestamps, [error[2] for error in errors], label=f'error_w')
ax_main.plot(timestamps, [error[3] for error in errors], label=f'error_h')



ax_main.set_xlabel('Time (s)')
ax_main.set_ylabel('Error')
ax_main.set_title('Error vs Time (Data sampled from CSV, 5 frames from video)')
ax_main.legend()
ax_main.grid(True)

# 视频处理部分
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video file at {video_path}")
    exit()

# 获取视频的总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 计算要提取的5个帧的位置
frame_positions = np.linspace(0, total_frames - 1, 5, dtype=int)

# 提取并插入指定位置的帧
ax_frames.axis('off')
for i, frame_pos in enumerate(frame_positions):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()
    if ret:
        # 计算当前帧对应的时间
        time = frame_pos / fps
        print(f"Successfully extracted frame at time: {time:.2f}s")
        
        # 在下方子图中显示视频帧
        ax_img = ax_frames.inset_axes([i*0.2, 0, 0.18, 1])
        ax_img.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax_img.axis('off')
    else:
        print(f"Failed to read frame at position: {frame_pos}")

cap.release()

plt.tight_layout()
plt.show()

print("视频帧提取和图表绘制完成")