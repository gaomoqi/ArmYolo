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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Record_Train")))
from JacobianModel import ImageJacobianNet, model

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
    """使用 YOLO 进行目标检测并计算特征点"""
    
    results = model(frame)
    obb_results = results[0].obb
    feature_points = []
    all_detections = []

    unique_classes = torch.unique(obb_results.cls).cpu().numpy()

    for cls_id in unique_classes:
        class_filter = obb_results.cls == cls_id
        class_confs = obb_results.conf[class_filter]
        class_boxes = obb_results.xyxyxyxy[class_filter]
        xywhr = obb_results.xywhr[class_filter]
        if len(class_confs) > 0:
            for i in range(len(class_confs)):
                conf = class_confs[i].item()
                box = class_boxes[i].cpu().numpy()
                xywhr[i][4] = xywhr[i][4] * (180 / np.pi)  # 角度转换
                all_detections.append((conf, box, int(cls_id), xywhr[i]))

    if not all_detections:
        return None
    best_conf, best_box, best_cls_id, best_xywhr = max(all_detections, key=lambda x: x[0])

    # 绘制边界框
    draw_bbox_with_depth(frame, best_box, f'Class {best_cls_id}', best_conf, best_xywhr)

    # 提取特征点
    points = np.array(best_box).reshape((-1, 1, 2)).astype(np.int32)
    x_coords, y_coords = points[:, 0, 0], points[:, 0, 1]

    # 计算中心点
    center_x, center_y = int(np.mean(x_coords)), int(np.mean(y_coords))
    # # 长边中点
    # width_mid_x, width_mid_y = int((x_coords[0] + x_coords[1]) / 2), int((y_coords[0] + y_coords[1]) / 2)
    # # 短边中点
    # height_mid_x, height_mid_y = int((x_coords[0] + x_coords[3]) / 2), int((y_coords[0] + y_coords[3]) / 2)
    
    width = ((x_coords[1]-x_coords[0])**2 + (y_coords[1]-y_coords[0])**2)**0.5
    height = ((x_coords[3]-x_coords[0])**2 + (y_coords[3]-y_coords[0])**2)**0.5
    # print("image_points",[center_x, center_y, width_mid_x, width_mid_y, height_mid_x, height_mid_y])
    print("image_info",[center_x, center_y, width, height])
    # 返回该目标的六个特征点数据
    return np.array([center_x, center_y, width, height])


def predict_velocity(model, velocities):
    """使用神经网络预测 UR5 末端速度"""
    if velocities.shape != (4,):
        return None
    normalized_input = input_scaler.transform(velocities.reshape(1, -1))

    input_tensor = torch.FloatTensor(normalized_input).unsqueeze(0)
    with torch.no_grad():
        predicted_velocity = model(input_tensor).numpy().flatten()

    denormalized_predicted_velocity = output_scaler.inverse_transform(predicted_velocity.reshape(1,-1)).flatten()

    return denormalized_predicted_velocity

def generate_urscript(v_cmd):
    # print("speed_cmd",v_cmd,"\n")
    v_cmd = [max(min(v_cmd[i], 0.02), -0.02) for i in range(3)]# 限定速度范围0.05
    v_cmd [3:6] = [0,0,0]
    ur_script = f"""
    speedl({v_cmd}, a=0.5, t=1)
    """
    return ur_script

def draw_bbox_with_depth(frame,  xyxy, label, conf, xywhr):
    points = np.array(xyxy).reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    
    x_coords, y_coords = points[:, 0, 0], points[:, 0, 1]
    center_x, center_y = int(np.mean(x_coords)), int(np.mean(y_coords))
    width = int(np.linalg.norm(points[0][0] - points[1][0]))
    height = int(np.linalg.norm(points[1][0] - points[2][0]))

    width_mid_x, width_mid_y = int((x_coords[0] + x_coords[1]) / 2), int((y_coords[0] + y_coords[1]) / 2)
    height_mid_x, height_mid_y = int((x_coords[0] + x_coords[3]) / 2), int((y_coords[0] + y_coords[3]) / 2)

    cv2.putText(frame, f'Center ({center_x}, {center_y})', (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(frame, f'Width: {width:.2f}', (width_mid_x, width_mid_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(frame, f'Height: {height:.2f}', (height_mid_x, height_mid_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
        cv2.putText(frame, f'P{idx+1} ({x}, {y})', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.circle(frame, (width_mid_x, width_mid_y), 5, (255, 255, 0), -1)
    cv2.circle(frame, (height_mid_x, height_mid_y), 5, (0, 255, 255), -1)
    
input_scaler = joblib.load('training_output2025_03_08_17_59/input_scaler.pkl')
output_scaler = joblib.load('training_output2025_03_08_17_59/output_scaler.pkl')
# velocity_model = JacobianModel.model
velocity_model = torch.load("training_output2025_03_08_17_59/jacobian_model_xyz.pt", map_location=torch.device('cpu'))
velocity_model.eval()

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
    target_position = None
    
    sock = connect_to_ur5(ur5_ip, port)

    yolo_model = YOLO("src/Record_Train/best.pt")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    while True:
        frame = get_camera_frame(pipeline)
        if frame is None:
            continue
        current_time = time.time()
        elapsed_time = current_time - start_time
        frame_number += 1
        feature_points = detect_feature_points(yolo_model, frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

        #识别不到时保持上一帧 识别到目标再跟踪
        if feature_points is None:
            continue

        if target_position is None:
            # target_position = list(map(int,input("请输入目标位置（以空格分隔）：").split()))
            # target_position = np.array([350, 294, 359, 436, 243, 301])#6输入
            target_position = np.array([537, 225, 155, 177])#4输入
            print(target_position)
            continue

        # print("feature_points", feature_points,"\n",target_position,"\n")
        error_cmd = feature_points - target_position
        error = np.linalg.norm(feature_points - target_position)# 计算误差

        # ----------------记录 error_cmd 和时间戳-------------------#
        error_log.write(f"{elapsed_time},{frame_number},{error_cmd[0]},{error_cmd[1]},{error_cmd[2]},{error_cmd[3]}\n")        # 在图像上添加时间戳和帧号
        cv2.putText(frame, f"Time: {elapsed_time:.3f}s Frame: {frame_number}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 保存带有时间戳的图像帧
        video_out.write(frame)

        # 记录 error_cmd、时间戳和帧号
        error_log.write(f"{elapsed_time:.3f},{frame_number},{','.join(map(str, error_cmd))}\n")
        error_log.flush()  # 确保数据立即写入文件
        #------------------------------------------------------------#

        # if error < error_threshold:
        #     print("目标已跟踪完成，暂停运动...")
        #     target_position = None
        #     continue
        
        image_speed_cmd = np.array([ - error_cmd[i] * 1.5 for i in range(4)])#u = -e*kp
        print("image_speed_cmd", image_speed_cmd,'\n')
        v_cmd = predict_velocity(velocity_model,image_speed_cmd)#根据图像误差预测输入速度
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
