from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import torch
import cv2

# Load the YOLO model
model = YOLO("src/best.pt")

# ---------------- Set up RealSense pipeline---------------------------#
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Enable depth stream

# # Start the pipeline
# pipeline.start(config)


#------------------setup my camera---------------#
cap=cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Create a VideoWriter object for output
# output_path = "./data/output_realsense_video.mp4"
output_path = "./data/output_capture_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))  # Set to match RealSense resolution
out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))  # Set to match  resolution


def draw_bbox_with_depth(frame, depth_frame, xyxy, label, conf,xywhr):
    """ Draw bounding box, label, coordinates, midpoints, depth, and print details """
    points = np.array(xyxy).reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Calculate bounding box properties
    x_coords, y_coords = points[:, 0, 0], points[:, 0, 1]
    center_x, center_y = int(np.mean(x_coords)), int(np.mean(y_coords))
    width = int(np.linalg.norm(points[0][0] - points[1][0]))  # Width
    height = int(np.linalg.norm(points[1][0] - points[2][0]))  # Height
    #theta = np.arctan2(y_coords[1] - y_coords[0], x_coords[1] - x_coords[0]) * (180 / np.pi)  # Rotation angle in degrees

    # Midpoints
    width_mid_x, width_mid_y = int((x_coords[0] + x_coords[1]) / 2), int((y_coords[0] + y_coords[1]) / 2)
    height_mid_x, height_mid_y = int((x_coords[0] + x_coords[3]) / 2), int((y_coords[0] + y_coords[3]) / 2)
    theta = np.arctan2(width_mid_y - center_y, center_x - width_mid_x) * (180 / np.pi)  # Rotation angle in degrees
    global last_valid_depth
    # Get depth at the center of the bounding box
    last_valid_depth=0.34
    depth = depth_frame.get_distance(center_x, center_y)
    if depth==0:
        if last_valid_depth is not None:
            depth =last_valid_depth
        else:
            depth=0.34
    else:
        last_valid_depth=depth
    # depth1 = depth_frame.get_distance(x_coords[3],y_coords[3])
    # depth2 = depth_frame.get_distance(x_coords[0],y_coords[0])
    # depth3 = depth_frame.get_distance(x_coords[2],y_coords[2])
    # depth4 = depth_frame.get_distance(x_coords[1],y_coords[1])
    # print(depth1,depth2,depth3,depth4)
    # Display properties on frame
    depth = 0

    cv2.putText(frame, f'Center ({center_x}, {center_y})', (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(frame, f'Depth: {depth:.2f}m', (center_x, center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f'Theta: {theta:.2f} deg', (center_x, center_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(frame, f'Width: {width:.2f}', (width_mid_x, width_mid_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(frame, f'Height: {height:.2f}', (height_mid_x, height_mid_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    #float(xywhr[0][4])
    # cv2.putText(frame, f'D1: {depth1:.2f}m', (x_coords[3],y_coords[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(frame, f'D2: {depth2:.2f}m', (x_coords[0],y_coords[0]+ 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(frame, f'D3: {depth3:.2f}m', (x_coords[2],y_coords[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(frame, f'D4: {depth4:.2f}m', (x_coords[1],y_coords[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display coordinates on image
    # for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
    #     cv2.putText(frame, f'P{idx+1} ({x}, {y})', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, f'P{1} ({x_coords[3]}, {y_coords[3]})', (x_coords[3], y_coords[3] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, f'P{2} ({x_coords[0]}, {y_coords[0]})', (x_coords[0], y_coords[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, f'P{3} ({x_coords[2]}, {y_coords[2]})', (x_coords[2], y_coords[2] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, f'P{4} ({x_coords[1]}, {y_coords[1]})', (x_coords[1], y_coords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    # Print all required information
    print(f"Center: ({center_x}, {center_y})")
    print(f"Width: {width}, Height: {height}")
    print(f"Rotation (theta): {theta:.2f} degrees")
    print(f"Depth: {depth:.2f} meters")
    print(f"Width Midpoint: ({width_mid_x}, {width_mid_y})")
    print(f"Height Midpoint: ({height_mid_x}, {height_mid_y})")
    
    # Draw key points
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Center in red
    cv2.circle(frame, (width_mid_x, width_mid_y), 5, (255, 255, 0), -1)  # Width midpoint in cyan
    cv2.circle(frame, (height_mid_x, height_mid_y), 5, (0, 255, 255), -1)  # Height midpoint in yellow
    return center_x, center_y, width, height, theta, depth, width_mid_x, width_mid_y, height_mid_x, height_mid_y,x_coords[3],y_coords[3],x_coords[0],y_coords[0],x_coords[2],y_coords[2],x_coords[1],y_coords[1]
# f1= open('./data/data2.txt','w+')

#----------------------Draw without depth------------------#
def draw_bbox_without_depth(frame, xyxy, label, conf,xywhr):
    """ Draw bounding box, label, coordinates, midpoints, depth, and print details """
    points = np.array(xyxy).reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Calculate bounding box properties
    x_coords, y_coords = points[:, 0, 0], points[:, 0, 1]
    center_x, center_y = int(np.mean(x_coords)), int(np.mean(y_coords))
    width = int(np.linalg.norm(points[0][0] - points[1][0]))  # Width
    height = int(np.linalg.norm(points[1][0] - points[2][0]))  # Height
    #theta = np.arctan2(y_coords[1] - y_coords[0], x_coords[1] - x_coords[0]) * (180 / np.pi)  # Rotation angle in degrees

    # Midpoints
    width_mid_x, width_mid_y = int((x_coords[0] + x_coords[1]) / 2), int((y_coords[0] + y_coords[1]) / 2)
    height_mid_x, height_mid_y = int((x_coords[0] + x_coords[3]) / 2), int((y_coords[0] + y_coords[3]) / 2)
    theta = np.arctan2(width_mid_y - center_y, center_x - width_mid_x) * (180 / np.pi)  # Rotation angle in degrees


    cv2.putText(frame, f'Center ({center_x}, {center_y})', (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    # cv2.putText(frame, f'Depth: {depth:.2f}m', (center_x, center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f'Theta: {theta:.2f} deg', (center_x, center_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(frame, f'Width: {width:.2f}', (width_mid_x, width_mid_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(frame, f'Height: {height:.2f}', (height_mid_x, height_mid_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    #float(xywhr[0][4])
    # cv2.putText(frame, f'D1: {depth1:.2f}m', (x_coords[3],y_coords[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(frame, f'D2: {depth2:.2f}m', (x_coords[0],y_coords[0]+ 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(frame, f'D3: {depth3:.2f}m', (x_coords[2],y_coords[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.putText(frame, f'D4: {depth4:.2f}m', (x_coords[1],y_coords[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display coordinates on image
    # for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
    #     cv2.putText(frame, f'P{idx+1} ({x}, {y})', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, f'P{1} ({x_coords[3]}, {y_coords[3]})', (x_coords[3], y_coords[3] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, f'P{2} ({x_coords[0]}, {y_coords[0]})', (x_coords[0], y_coords[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, f'P{3} ({x_coords[2]}, {y_coords[2]})', (x_coords[2], y_coords[2] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, f'P{4} ({x_coords[1]}, {y_coords[1]})', (x_coords[1], y_coords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    # Print all required information
    print(f"Center: ({center_x}, {center_y})")
    print(f"Width: {width}, Height: {height}")
    print(f"Rotation (theta): {theta:.2f} degrees")
    print(f"Width Midpoint: ({width_mid_x}, {width_mid_y})")
    print(f"Height Midpoint: ({height_mid_x}, {height_mid_y})")
    
    # Draw key points
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Center in red
    cv2.circle(frame, (width_mid_x, width_mid_y), 5, (255, 255, 0), -1)  # Width midpoint in cyan
    cv2.circle(frame, (height_mid_x, height_mid_y), 5, (0, 255, 255), -1)  # Height midpoint in yellow
    return center_x, center_y, width, height, theta,  width_mid_x, width_mid_y, height_mid_x, height_mid_y,x_coords[3],y_coords[3],x_coords[0],y_coords[0],x_coords[2],y_coords[2],x_coords[1],y_coords[1]


try:
    while True:
        f = open('./data/data.txt', 'w+')
        # ----------Wait for coherent frames from RealSense------------#
        # frames = pipeline.wait_for_frames()
        # color_frame = frames.get_color_frame()
        # depth_frame = frames.get_depth_frame()
        # if not color_frame or not depth_frame:
        #     continue

        # # Convert RealSense color frame to a NumPy array
        # frame = np.asanyarray(color_frame.get_data())

        #----------------Get frames from my camera---------------#
        _, frame = cap.read()
    
        # Run YOLO inference on the RealSense frame
        results = model(frame)

        # Extract OBB results
        obb_results = results[0].obb

        # Get class-wise detections (choose the highest confidence detection per class)
        unique_classes = torch.unique(obb_results.cls).cpu().numpy()  # Get unique class IDs
        for cls_id in unique_classes:
            # Filter out detections for the current class
            class_filter = obb_results.cls == cls_id
            class_confs = obb_results.conf[class_filter]
            class_boxes = obb_results.xyxyxyxy[class_filter]
            xywhr = obb_results.xywhr[class_filter]
            xywhr[0][4]=xywhr[0][4]* (180 / np.pi)
            print(xywhr)
            # Select the detection with the highest confidence
            if len(class_confs) > 0:
                best_idx = torch.argmax(class_confs)
                best_conf = class_confs[best_idx].item()
                best_box = class_boxes[best_idx].cpu().numpy()

                # Draw bounding box, label, coordinates, midpoints, and depth on the frame
                # xc, yc, w, h, theta, depth,width_mid_x, width_mid_y, height_mid_x, height_mid_y, x1_x, x1_y, x2_x, x2_y,x3_x, x3_y, x4_x, x4_y =draw_bbox_with_depth(frame, depth_frame, best_box, f'Class {int(cls_id)}', best_conf,xywhr)
                xc, yc, w, h, theta, width_mid_x, width_mid_y, height_mid_x, height_mid_y, x1_x, x1_y, x2_x, x2_y,x3_x, x3_y, x4_x, x4_y =draw_bbox_without_depth(frame, best_box, f'Class {int(cls_id)}', best_conf,xywhr)
                # if xc != 0:
                #     f.write(str(xc) + '\t' + str(yc) + '\t' + str(w) + '\t' + str(h) + '\t' + str(theta) + '\t' + str(
                #         depth) + '\t' + str(width_mid_x) + '\t' + str(width_mid_y) + '\t' + str(height_mid_x) + '\t' + str(
                #         height_mid_y)+ '\t'+ str(x1_x) + '\t' + str(x1_y) + '\t' + str(x2_x) + '\t' + str(
                #         x2_y) + '\t'+ str(x3_x) + '\t' + str(x3_y) + '\t' + str(x4_x) + '\t' + str(
                #         x4_y)+ '\n')

                # f.write(str(xc) + '\t' + str(yc) + '\t' + str(w) + '\t' + str(h) + '\t' + str(theta) + '\t' + str(
                #     depth) + '\t' + str(width_mid_x) + '\t' + str(width_mid_y) + '\t' + str(height_mid_x) + '\t' + str(
                #     height_mid_y) + '\t' + str(x1_x) + '\t' + str(x1_y) + '\t' + str(x2_x) + '\t' + str(
                #     x2_y) + '\t' + str(x3_x) + '\t' + str(x3_y) + '\t' + str(x4_x) + '\t' + str(
                #     x4_y) + '\n')

                f.write(str(xc) + '\t' + str(yc) + '\t' + str(w) + '\t' + str(h) + '\t' + str(theta) + '\t' + str(
                    'depth unkown') + '\t' + str(width_mid_x) + '\t' + str(width_mid_y) + '\t' + str(height_mid_x) + '\t' + str(
                    height_mid_y) + '\t' + str(x1_x) + '\t' + str(x1_y) + '\t' + str(x2_x) + '\t' + str(
                    x2_y) + '\t' + str(x3_x) + '\t' + str(x3_y) + '\t' + str(x4_x) + '\t' + str(
                    x4_y) + '\n')

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow("YOLOv11 OBB RealSense Inference with Depth", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline and release resources
    # pipeline.stop()
    cap.close()
    out.release()
    cv2.destroyAllWindows()
