#writen by GaoMoqi  record data from camera and arm
import time
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import torch
import cv2
import os
from datetime import datetime
import csv
import recordarm as ra


def draw_bbox_with_depth(frame, bbox, track_id, conf, depth):
    x_min, y_min, x_max, y_max = map(int, bbox)
    color = (0, 255, 0)  # 绿色边界框
    text_color = (0, 0, 255)  # 红色字体
    thickness = 2

    # 绘制边界框
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

    # 计算特征点
    center_x, center_y =int((x_min + x_max) // 2), int((y_min + y_max) // 2)
    width_mid_x, width_mid_y = center_x, y_max
    height_mid_x, height_mid_y = x_min, center_y
    # 计算边界框的宽度和高度
    width = int(x_max - x_min)
    height = int(y_max - y_min)

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

# 确保输出目录存在
output_dir = "/home/gaomoqi/ArmYolo_ws/original_data/data_2025_3_18"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
# Load the YOLO model
model = YOLO("./src/Record/yolov8n.pt")

# Set up RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Create a VideoWriter object for output
output_path = os.path.join(output_dir, f"output_realsense_video_{timestamp}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))



if __name__ == "__main__":
    try:
        icsvfile = open(os.path.join(output_dir, f"image_data_{timestamp}.csv"), 'w+', newline='')
        icsv_writer = csv.writer(icsvfile)
        icsv_writer.writerow(['xc', 'yc', 'w', 'h', 'theta', 'depth', 'width_mid_x', 'width_mid_y', 
                                'height_mid_x', 'height_mid_y', 'x1_x', 'x1_y', 'x2_x', 'x2_y', 
                                'x3_x', 'x3_y', 'x4_x', 'x4_y', 'sample_time'])
        
        rcsvfile = open(os.path.join(output_dir, f"robot_data_{timestamp}.csv"), 'w+', newline='')
        if ra.args.binary:
            rcsv_writer = ra.csv_binary_writer.CSVBinaryWriter(rcsvfile, ra.output_names, ra.output_types)
        else:
            rcsv_writer = ra.csv_writer.CSVWriter(rcsvfile, ra.output_names, ra.output_types)

        rcsv_writer.writeheader()
        start_time = time.time()  # 记录开始时间
        last_record_time = start_time  # 添加上次记录时间

        frame_count = 0  # 添加帧计数器
        i = 0
        image_success = False
        while True:
            
            time.sleep(0.01)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())

            """使用 YOLO 进行目标跟踪，并计算特征点"""
            results = model.track(frame, persist=True, classes = [41], device = "cuda")  # 使用 track 进行目标跟踪
            if not results[0].boxes:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()  # 获取边界框
            track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else np.zeros(len(boxes))  # 获取目标 ID
            feature_points = []
            all_detections = []

            for i, box in enumerate(boxes):
                track_id = int(track_ids[i])  # 目标的跟踪 ID
                conf = results[0].boxes.conf[i].item()  # 置信度
                all_detections.append((conf, box, track_id))
                
            current_time = time.time()
            sample_time = current_time - start_time
            # 选取置信度最高的目标
            best_conf, best_box, best_track_id = max(all_detections, key=lambda x: x[0])

            """ 计算特征点['xc', 'yc', 'w', 'h', 'theta', 'depth', 'width_mid_x', 'width_mid_y', 
                       'height_mid_x', 'height_mid_y', 'x1_x', 'x1_y', 'x2_x', 'x2_y', 
                       'x3_x', 'x3_y', 'x4_x', 'x4_y', 'sample_time']"""
            # 计算特征点
            x_min, y_min, x_max, y_max = map(int, best_box)
            xc, yc = (x_min + x_max) // 2, (y_min + y_max) // 2
            w, h = x_max - x_min, y_max - y_min
            theta = 0  # Assuming no rotation for simplicity
            depth = depth_frame.get_distance(int(xc), int(yc))
            width_mid_x, width_mid_y = (x_min + x_max) / 2, y_max
            height_mid_x, height_mid_y = x_min, (y_min + y_max) / 2

            # Assuming the bounding box is axis-aligned, the corners are:
            x1_x, x1_y = x_min, y_min
            x2_x, x2_y = x_min, y_max
            x3_x, x3_y = x_max, y_min
            x4_x, x4_y = x_max, y_max
            
            draw_bbox_with_depth(frame, best_box, f"ID {best_track_id}", best_conf, None)



            if current_time - last_record_time > 0.05:# 每_秒记录一次
                icsv_writer.writerow([xc, yc, w, h, theta, depth, width_mid_x, width_mid_y, 
                            height_mid_x, height_mid_y, x1_x, x1_y, x2_x, x2_y, 
                            x3_x, x3_y, x4_x, x4_y,sample_time])
                last_record_time = current_time
                start_time = time.time()# 重置计时
                image_success = True
            
                    
            if ra.args.samples > 0:
                ra.sys.stdout.write("\r")
                ra.sys.stdout.write("{:.2%} done.".format(float(i) / float(ra.args.samples)))
                ra.sys.stdout.flush()
            else:
                ra.sys.stdout.write("\r")
                ra.sys.stdout.write("{:3d} samples.".format(i))
                ra.sys.stdout.flush()

            if ra.args.buffered:
                state = ra.con.receive_buffered(ra.args.binary)
            else:
                state = ra.con.receive(ra.args.binary)
            if state is not None and image_success:
                rcsv_writer.writerow(state)


            #记录频率    
            
            out.write(frame)
            cv2.imshow("YOLOv11 OBB RealSense Inference with Depth", frame)
            i += 1
            image_success = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        out.release
        cv2.destroyAllWindows()
