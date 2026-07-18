import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Create base dir
base_dir = "dataset_v2"
classes = {1: "PET", 2: "HDPE", 3: "PP", 4: "LDPE", 5: "MISC", 6: "BACKGROUND"}
for c in classes.values():
    os.makedirs(os.path.join(base_dir, c), exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

current_class = "PET"
auto_capture = False
frame_count = 0

print("Controls: 1-6 to select class, 'a' to toggle auto-capture, 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        
        color_image = np.asanyarray(color_frame.get_data())
        display_img = color_image.copy()
        
        # Display Info
        cv2.putText(display_img, f"Class: {current_class}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if auto_capture:
            cv2.rectangle(display_img, (0, 0), (640, 480), (0, 255, 0), 4)
            if frame_count % 15 == 0: # 2 frames per second
                # Ensure unique filenames for resume mode
                existing = len(os.listdir(os.path.join(base_dir, current_class)))
                filename = os.path.join(base_dir, current_class, f"{current_class}_{existing}.jpg")
                cv2.imwrite(filename, color_image)
        
        cv2.imshow("SmartSeg Capture", display_img)
        key = cv2.waitKey(1)
        
        if key == ord('q'): break
        elif key == ord('a'): auto_capture = not auto_capture
        elif key in [ord(str(i)) for i in range(1, 7)]:
            current_class = classes[int(chr(key))]

        frame_count += 1
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
