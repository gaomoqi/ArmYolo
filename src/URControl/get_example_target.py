#writen by GaoMoqi used to test yolo on depth capture
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import torch
import cv2
import os
from datetime import datetime
import csv
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
    width = x_max - x_min
    height = y_max - y_min

    print(f"Track ID {best_track_id} | image_info:", [center_x, center_y, width, height])

    # 绘制目标框、ID 和特征点
    draw_bbox_with_depth(frame, best_box, f"ID {best_track_id}", best_conf, None)

    return np.array([center_x, center_y, width_mid_x, width_mid_y, height_mid_x, height_mid_y, center_x, center_y, width, height])

def draw_bbox_with_depth(frame, bbox, track_id, conf, depth):
    x_min, y_min, x_max, y_max = map(int, bbox)
    color = (0, 255, 0)  # 绿色边界框
    text_color = (0, 0, 255)  # 红色字体
    thickness = 2

    # 绘制边界框
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

    # 计算特征点
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
    width_mid_x, width_mid_y = center_x, y_max
    height_mid_x, height_mid_y = x_min, center_y
    # 计算边界框的宽度和高度
    width = x_max - x_min
    height = y_max - y_min

    # 标注特征点
    feature_color = (255, 0, 0)  # 蓝色点
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

    # 置信度 + 目标 ID
    label = f"ID {track_id} | Conf: {conf:.2f}"
    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    
# Load the YOLO model
model = YOLO("src/Record/yolov8n.pt")

# Set up RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)


# Start the pipeline
pipeline.start(config)

try:
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        detect_feature_points(model, frame)
        cv2.imshow("YOLOv8 OBB RealSense Inference with Depth", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
