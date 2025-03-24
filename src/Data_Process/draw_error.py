import matplotlib.pyplot as plt
import cv2
import numpy as np
import csv
import os

path = "record_error/record_error_new_5"
error_filename = os.path.join(path, "error_log.csv")
v_cmd_filename = os.path.join(path, "v_cmd_log.csv")
video_path = os.path.join(path, "output.mp4")

def read_csv_data(filename):
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # 跳过标题行
        for row in csvreader:
            try:
                data.append([float(value) for value in row])
            except ValueError:
                continue
    return data

error_data = read_csv_data(error_filename)
v_cmd_data = read_csv_data(v_cmd_filename)

# 每隔5个点取一个点
error_data_sampled = error_data[::5]
v_cmd_data_sampled = v_cmd_data[::5]

# 提取时间戳和数据
error_timestamps = [row[0] for row in error_data_sampled]
error_values = [row[2:] for row in error_data_sampled]

v_cmd_timestamps = [row[0] for row in v_cmd_data_sampled]
v_cmd_values = [row[2:5] for row in v_cmd_data_sampled]  # 只取xyz三个维度

# 创建两幅图
fig1, (ax_error, ax_frames) = plt.subplots(2, 1, figsize=(20, 16), height_ratios=[3, 1])
fig2, ax_v_cmd = plt.subplots(figsize=(20, 10))

# 绘制误差曲线
for i, label in enumerate(['error_x', 'error_y', 'error_w', 'error_h']):
    ax_error.plot(error_timestamps, [error[i] for error in error_values], label=label)

ax_error.set_xlabel('Time (s)')
ax_error.set_ylabel('Error')
ax_error.set_title('Error vs Time (Data sampled from CSV, 5 frames from video)')
ax_error.legend()
ax_error.grid(True)

# 绘制v_cmd曲线（三维：xyz）
for i, label in enumerate(['v_cmd_x', 'v_cmd_y', 'v_cmd_z']):
    ax_v_cmd.plot(v_cmd_timestamps, [v_cmd[i] for v_cmd in v_cmd_values], label=label)

ax_v_cmd.set_xlabel('Time (s)')
ax_v_cmd.set_ylabel('V_cmd')
ax_v_cmd.set_title('V_cmd vs Time (Data sampled from CSV, xyz dimensions)')
ax_v_cmd.legend()
ax_v_cmd.grid(True)

# 视频处理部分
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video file at {video_path}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

frame_positions = np.linspace(0, total_frames - 1, 5, dtype=int)

ax_frames.axis('off')
for i, frame_pos in enumerate(frame_positions):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()
    if ret:
        time = frame_pos / fps
        print(f"Successfully extracted frame at time: {time:.2f}s")
        
        ax_img = ax_frames.inset_axes([i*0.2, 0, 0.18, 1])
        ax_img.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax_img.axis('off')
    else:
        print(f"Failed to read frame at position: {frame_pos}")

# 保存图片
output_image_path1 = os.path.join(path, "error_plot_with_frames.png")
output_image_path2 = os.path.join(path, "v_cmd_plot.png")

fig1.savefig(output_image_path1)
fig2.savefig(output_image_path2)

print(f"Error图表已保存到 {output_image_path1}")
print(f"V_cmd图表已保存到 {output_image_path2}")

cap.release()
plt.tight_layout()
plt.show()

print("视频帧提取和图表绘制完成")