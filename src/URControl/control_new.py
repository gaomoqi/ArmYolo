import socket
import sys
import time
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from ultralytics import YOLO
import joblib
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Train_new")))
from JacobianModel_new import ImageJacobianNet, model
print(model)
def connect_to_ur5(ip, port):
    """建立与 UR5 机器人的 TCP 连接"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    return sock

def get_camera_frame(pipeline):
    """获取 RealSense 相机的 RGB 帧"""
    
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data())

def detect_feature_points(model, frame):
    """使用 YOLO 进行目标跟踪，并计算特征点"""
    results = model.track(frame, persist=True, classes = [41], device = "cuda")  # 使用 track 进行目标跟踪
    if not results[0].boxes:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()  # 获取边界框
    track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else np.zeros(len(boxes))  # 获取目标 ID
    feature_points = []
    all_detections = []

    for i, box in enumerate(boxes):
        track_id = int(track_ids[i])  # 目标的跟踪 ID
        conf = results[0].boxes.conf[i].item()  # 置信度
        all_detections.append((conf, box, track_id))

    if not all_detections:
        return None

    # 选取置信度最高的目标
    best_conf, best_box, best_track_id = max(all_detections, key=lambda x: x[0])

    # 计算特征点
    x_min, y_min, x_max, y_max = best_box
    center_x, center_y = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
    width_mid_x, width_mid_y = int((x_min + x_max) / 2), int(y_max)
    height_mid_x, height_mid_y = int(x_min), int((y_min + y_max) / 2)
    width = int(x_max - x_min)
    height = int(y_max - y_min)

    print(f"Track ID {best_track_id} | image_info:", [center_x, center_y, width, height])

    # 绘制目标框、ID 和特征点
    draw_bbox_with_depth(frame, best_box, f"ID {best_track_id}", best_conf, None)

    return np.array([center_x, center_y, width_mid_x, width_mid_y, height_mid_x, height_mid_y, center_x, center_y, width, height])

def predict_jacobian(model, position, tcp_velocity = np.array([0.01,0.01,0.01])):
    if position.shape != (6,) or tcp_velocity.shape != (3,):
        return None
    # 归一化输入数据
    
    position = position_scaler.transform(position.reshape(1, -1)).flatten()
    # tcp_velocity = tcp_velocity_scaler.transform(tcp_velocity.reshape(1, -1)).flatten()

    position_tensor = torch.tensor(position).float().unsqueeze(0)
    tcp_velocity_tensor = torch.tensor(tcp_velocity).float().unsqueeze(0)
    with torch.no_grad():
        output = model(position_tensor, tcp_velocity_tensor)
        jacobian = output[1]
    return jacobian  # 4x3

def generate_urscript(v_cmd):
    # print("speed_cmd",v_cmd,"\n")
    v_cmd = [max(min(v_cmd[i], 0.02), -0.02) for i in range(3)]# 限定速度范围0.05
    v_cmd [3:6] = [0,0,0]
    ur_script = f"""
    speedl({v_cmd}, a=0.1, t=1)
    """
    return ur_script

def draw_bbox_with_depth(frame, bbox, track_id, conf, depth):
    x_min, y_min, x_max, y_max = map(int, bbox)
    color = (0, 255, 0)  # 绿色边界框
    text_color = (0, 255, 0)  # 绿色文字
    thickness = 2

    # 绘制边界框
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
    if target_position is not None:
        target_x, target_y, target_width, target_height = target_position
        target_x_min = int(target_x - target_width / 2)
        target_x_max = int(target_x + target_width / 2)
        target_y_min = int(target_y - target_height / 2)
        target_y_max = int(target_y + target_height / 2)
        target_color = (0, 0, 255) 
        cv2.circle(frame, (target_x, target_y), 5, target_color, -1)

        cv2.rectangle(frame, (target_x_min, target_y_min), (target_x_max, target_y_max), target_color, thickness)
    # 计算特征点
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
    width_mid_x, width_mid_y = center_x, y_max
    height_mid_x, height_mid_y = x_min, center_y
    # 计算边界框的宽度和高度
    width = x_max - x_min
    height = y_max - y_min

    # 标注特征点
    feature_color = (0, 255, 0)  # 绿色特征点
    radius = 5
    cv2.circle(frame, (center_x, center_y), radius, feature_color, -1)
    cv2.circle(frame, (width_mid_x, width_mid_y), radius, feature_color, -1)
    cv2.circle(frame, (height_mid_x, height_mid_y), radius, feature_color, -1)

    # 打印特征点坐标和边界框尺寸
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1

    # 中心点坐标
    cv2.putText(frame, f"({center_x},{center_y})", (center_x + 5, center_y - 5), font, font_scale, text_color, font_thickness)

    # 宽度中点坐标
    cv2.putText(frame, f"({width_mid_x},{width_mid_y})", (width_mid_x + 5, width_mid_y - 5), font, font_scale, text_color, font_thickness)

    # 高度中点坐标
    cv2.putText(frame, f"({height_mid_x},{height_mid_y})", (height_mid_x + 5, height_mid_y + 15), font, font_scale, text_color, font_thickness)

    # 边界框尺寸
    cv2.putText(frame, f"{width}x{height}", (x_min + 5, y_min + 20), font, font_scale, text_color, font_thickness)

    # # 置信度 + 目标 ID
    # label = f"ID {track_id} | Conf: {conf:.2f}"
    # cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    
position_scaler = joblib.load('training_new/training_output_new_12/position_scaler.pkl') 
# tcp_velocity_scaler = joblib.load('training_new/training_output_new_26/tcp_velocity_scaler.pkl')    
# uvwh_scaler = joblib.load('training_new/training_output_new_26/output_scaler.pkl')
jacbobian_model = torch.load("training_new/training_output_new_12/final_model.pt", map_location=torch.device('cpu'))
jacbobian_model.eval()
target_position = np.array([531, 293, 214, 250])  # 4输入

def main():
    # #----------记录数据---------#
    data_dir = f"record_error_{time.strftime('%Y_%m_%d_%H_%M')}"
    os.makedirs(data_dir, exist_ok=True)
    
    # 使用 mp4v 编解码器，创建 .mp4 文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(os.path.join(data_dir, 'output.mp4'), fourcc, 30.0, (640, 480))

    # 打开 CSV 文件用于记录 error_cmd 和时间戳
    error_log = open(os.path.join(data_dir, 'error_log.csv'), 'w')
    error_log.write("timestamp,frame_number,error_cmd_x1,error_cmd_y1,error_cmd_x2,error_cmd_y2,error_cmd_x3,error_cmd_y3\n")

    frame_number = 0
    start_time = time.time()
    #-----------------------------------#

    ur5_ip = "10.149.230.1"
    port = 30002
    error_threshold = 10  # 误差阈值
    
    sock = connect_to_ur5(ur5_ip, port)

    yolo_model = YOLO("src/Record/yolov8n.pt").to('cuda')
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    frame_count = 0
    feature_points_list = []

    while True:
        frame = get_camera_frame(pipeline)
        if frame is None:
            continue
        current_time = time.time()
        elapsed_time = current_time - start_time
        frame_number += 1
        feature_points = detect_feature_points(yolo_model, frame)
        if feature_points is None:
            continue

        feature_points_list.append(feature_points)
        frame_count += 1

        if frame_count < 3:
            continue

        # 计算特征点的平均值
        avg_feature_points = np.mean(feature_points_list, axis=0)
        feature_points_list = []
        frame_count = 0

        positions_xy = avg_feature_points[:6]
        features = avg_feature_points[6:]
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

        error_cmd = features - target_position
        error = np.linalg.norm(features - target_position)  # 计算误差

        # 记录 error_cmd 和时间戳
        error_log.write(f"{elapsed_time},{frame_number},{error_cmd[0]},{error_cmd[1]},{error_cmd[2]},{error_cmd[3]}\n")
        cv2.putText(frame, f"Time: {elapsed_time:.3f}s Frame: {frame_number}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        video_out.write(frame)
        error_log.flush()  # 确保数据立即写入文件

        image_speed_cmd = np.array([-error_cmd[i] * 1 for i in range(4)])  # u = -e*kp
        print("image_speed_cmd", image_speed_cmd, '\n')
        fake_TCP_velocity = np.array([0.01, 0.01, 0.01])  # 虚拟TCP速度
        jacobian_matrix = predict_jacobian(jacbobian_model, positions_xy, fake_TCP_velocity)  # 根据位置信息预测雅可比矩阵
        jacobian_pinv = np.linalg.pinv(jacobian_matrix)  # 计算 (3, 4) 形状的伪逆
        # image_speed_cmd = uvwh_scaler.transform(image_speed_cmd.reshape(1, -1)).flatten()  # 归一化uvwh数据
        v_cmd = np.dot(jacobian_pinv, image_speed_cmd).flatten()  # 计算 (3, 4) @ (4,) -> (3,)
        # v_cmd = tcp_velocity_scaler.inverse_transform(v_cmd.reshape(1, -1)).flatten()  # 反归一化v_cmd
        ur_script = generate_urscript(v_cmd)
        print("URScript:", ur_script)
        sock.sendall(ur_script.encode('utf-8'))

        time.sleep(0.1)

    # error_log.close()
    # video_out.release()
    cv2.destroyAllWindows()
    pipeline.stop()

if __name__ == "__main__":
    main()
