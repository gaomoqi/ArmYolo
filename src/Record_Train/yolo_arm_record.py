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

# 确保输出目录存在
output_dir = "/home/gaomoqi/ArmYolo_ws/data_2025_1_20"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
# Load the YOLO model
model = YOLO("./src/best.pt")

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

def draw_bbox_with_depth(frame, depth_frame, xyxy, label, conf, xywhr):
    points = np.array(xyxy).reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    
    x_coords, y_coords = points[:, 0, 0], points[:, 0, 1]
    center_x, center_y = int(np.mean(x_coords)), int(np.mean(y_coords))
    width = int(np.linalg.norm(points[0][0] - points[1][0]))
    height = int(np.linalg.norm(points[1][0] - points[2][0]))

    width_mid_x, width_mid_y = int((x_coords[0] + x_coords[1]) / 2), int((y_coords[0] + y_coords[1]) / 2)
    height_mid_x, height_mid_y = int((x_coords[0] + x_coords[3]) / 2), int((y_coords[0] + y_coords[3]) / 2)
    theta = np.arctan2(width_mid_y - center_y, center_x - width_mid_x) * (180 / np.pi)

    depth = depth_frame.get_distance(center_x, center_y)

    cv2.putText(frame, f'Center ({center_x}, {center_y})', (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(frame, f'Depth: {depth:.2f}m', (center_x, center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f'Theta: {theta:.2f} deg', (center_x, center_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(frame, f'Width: {width:.2f}', (width_mid_x, width_mid_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(frame, f'Height: {height:.2f}', (height_mid_x, height_mid_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
        cv2.putText(frame, f'P{idx+1} ({x}, {y})', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.circle(frame, (width_mid_x, width_mid_y), 5, (255, 255, 0), -1)
    cv2.circle(frame, (height_mid_x, height_mid_y), 5, (0, 255, 255), -1)

    return center_x, center_y, width, height, theta, depth, width_mid_x, width_mid_y, height_mid_x, height_mid_y, x_coords[3], y_coords[3], x_coords[0], y_coords[0], x_coords[2], y_coords[2], x_coords[1], y_coords[1]


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
            results = model(frame)
            obb_results = results[0].obb

            unique_classes = torch.unique(obb_results.cls).cpu().numpy()
            for cls_id in unique_classes:
                class_filter = obb_results.cls == cls_id
                class_confs = obb_results.conf[class_filter]
                class_boxes = obb_results.xyxyxyxy[class_filter]
                xywhr = obb_results.xywhr[class_filter]
                xywhr[0][4] = xywhr[0][4] * (180 / np.pi)

                if len(class_confs) > 0:
                    current_time = time.time()
                    sample_time = current_time - start_time
                    best_idx = torch.argmax(class_confs)
                    best_conf = class_confs[best_idx].item()
                    best_box = class_boxes[best_idx].cpu().numpy()
                    xc, yc, w, h, theta, depth, width_mid_x, width_mid_y, height_mid_x, height_mid_y, x1_x, x1_y, x2_x, x2_y, x3_x, x3_y, x4_x, x4_y = draw_bbox_with_depth(frame, depth_frame, best_box, f'Class {int(cls_id)}', best_conf, xywhr)
                    icsv_writer.writerow([xc, yc, w, h, theta, depth, width_mid_x, width_mid_y, 
                                        height_mid_x, height_mid_y, x1_x, x1_y, x2_x, x2_y, 
                                        x3_x, x3_y, x4_x, x4_y,sample_time])
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
