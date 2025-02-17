import socket
import time
import numpy as np
import torch
import pyrealsense2 as rs
from ultralytics import YOLO

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
    """使用 YOLO 进行目标检测并计算特征点"""
    results = model(frame)
    obb_results = results[0].obb
    feature_points = []
    unique_classes = torch.unique(obb_results.cls).cpu().numpy()
    for cls_id in unique_classes:
        class_filter = obb_results.cls == cls_id
        class_confs = obb_results.conf[class_filter]
        class_boxes = obb_results.xyxyxyxy[class_filter]
        xywhr = obb_results.xywhr[class_filter]
        xywhr[0][4] = xywhr[0][4] * (180 / np.pi)

        if len(class_confs) > 0:
            best_idx = torch.argmax(class_confs)
            best_box = class_boxes[best_idx].cpu().numpy()
            x1, y1, x2, y2, x3, y3, x4, y4 = best_box
            # 计算中心点
            center_x = int((x1 + x2 + x3 + x4) / 4)
            center_y = int((y1 + y2 + y3 + y4) / 4)
            # 长边中点
            width_mid_x = int((x1 + x2) / 2)
            width_mid_y = int((y1 + y2) / 2)  
            # 短边中点
            height_mid_x = int((x3 + x4) / 2)
            height_mid_y = int((y3 + y4) / 2)
           
            feature_points.append([center_x, center_y, width_mid_x, width_mid_y, height_mid_x, height_mid_y])
    return np.array(feature_points)

def predict_velocity(model, prev_points, curr_points, dt):
    """使用神经网络预测 UR5 末端速度"""
    if prev_points is None or len(prev_points) == 0:
        return np.zeros(6)
    velocities = (curr_points - prev_points) / dt
    input_tensor = torch.tensor(velocities).float().unsqueeze(0)
    with torch.no_grad():
        predicted_velocity = model(input_tensor).numpy().flatten()
    
    return predicted_velocity

def generate_urscript(velocity, error):
    if any(v > 100 for v in velocity):
        print("识别速度过大,输入速度为0")
        speed_cmd = [0.0] * 6  # 设置所有速度为 0

        ur_script = f"""
        speedl({speed_cmd}, a=0.01, t=1)
        """
        return ur_script
    
    else:
        speed_cmd = [velocity[i] - error[i] * 0.01 for i in range(3)] + [velocity[i] - error[i] * 0.01 for i in range(3,6)]
        speed_cmd = [ max(min(speed_cmd[i], 0.1), -0.1) for i in range(6)]# 限定速度范围0.1
        
        ur_script = f"""
        speedl({speed_cmd}, a=0.01, t=1)
        """
        return ur_script

def main():
    ur5_ip = "192.168.1.100"
    port = 30002
    error_threshold = 10  # 误差阈值
    target_position = None
    
    sock = connect_to_ur5(ur5_ip, port)
    yolo_model = YOLO("src/Record_Train/best.pt")
    velocity_model = torch.load("training_output2025_01_21_15_59/jacobian_model.pt")
    velocity_model.eval()
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    prev_positions = None
    prev_time = time.time()
    
    while True:
        frame = get_camera_frame(pipeline)
        if frame is None:
            continue
        
        feature_points = detect_feature_points(yolo_model, frame)
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        
        if target_position is None:
            target_position = input("请输入目标位置（以空格分隔）：").split()
            continue
        
        error = np.linalg.norm(feature_points - target_position)# 计算误差
        if error < error_threshold:
            print("目标已跟踪完成，暂停运动...")
            target_position = None
            continue
        
        velocity = predict_velocity(velocity_model, prev_positions, feature_points, dt)
        
        ur_script = generate_urscript(velocity, error)
        print("URScript:", ur_script)
        #sock.sendall(ur_script.encode('utf-8'))
        
        prev_positions = feature_points
        time.sleep(0.1)

if __name__ == "__main__":
    main()
