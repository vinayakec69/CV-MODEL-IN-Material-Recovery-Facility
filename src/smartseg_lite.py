import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import collections

print("Loading YOLOv8 Classification model...")
model = YOLO('../weights/best.pt')

# 1. RealSense Setup (NO EXPENSIVE ALIGNMENT = NO CRASHES!)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 2. UPGRADED Majority Voting Buffer (10 frames instead of 5)
history = collections.deque(maxlen=10)
SUPERMAJORITY = 7  # 7 out of 10 frames must agree!

print("\nCamera started! Waiting for objects to pass under camera...")

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
            
        color_image = np.asanyarray(color_frame.get_data())
        
        # 3. LIGHTWEIGHT DEPTH GATE
        dist = depth_frame.get_distance(320, 240)
        
        if 0 < dist < 0.7:
            # Object detected in range! Run YOLO classification.
            results = model(color_image, verbose=False)
            
            # Extract the classification probabilities
            probs = results[0].probs
            top1_index = probs.top1
            top1_conf = float(probs.top1conf)
            class_name = model.names[top1_index]
            
            # Only trust predictions with > 75% confidence
            if top1_conf > 0.75:
                history.append(class_name)
                
                # Show live confidence on screen
                cv2.putText(color_image, f"Raw: {class_name} ({top1_conf*100:.1f}%)", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Check supermajority
                if len(history) >= SUPERMAJORITY:
                    final_class = max(set(history), key=history.count)
                    vote_count = history.count(final_class)
                    
                    if vote_count >= SUPERMAJORITY:
                        cv2.putText(color_image, f"DETECTED: {final_class}", (20, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    else:
                        cv2.putText(color_image, f"Stabilizing... ({vote_count}/{SUPERMAJORITY})", (20, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(color_image, f"Low conf: {top1_conf*100:.1f}%", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Belt is empty
            history.clear()
            cv2.putText(color_image, "Belt Empty...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
        cv2.imshow('SmartSeg Live', color_image)
        
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Clean exit.")
