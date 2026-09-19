#!/usr/bin/env python3
"""
AI SORTING SYSTEM v2.3 - DEEP MATH EDITION
=============================================
1. Auto-calibrates mm/pixel from depth sensor at startup
2. Linear regression velocity (uses ALL data points, immune to jitter)
3. R-squared confidence metric
4. Classes: PET, HDPE, PP

Belt: BOTTOM -> TOP in camera. Servo past the top.
Measured: 388mm from commit line (screen center) to servo.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import Jetson.GPIO as GPIO
from ultralytics import YOLO
from collections import deque

# ==================== HARDWARE ====================
BASE_SERVO_PIN  = 15
STACK_SERVO_PIN = 18
BASE_HOME       = 90
STACK_HOME      = 0
STACK_SWEEP     = 130

# ==================== SORTING ====================
BIN_ANGLES    = {"PET": 45, "HDPE": 135, "PP": 60, "PS": 120}
VALID_CLASSES = ["PET", "HDPE", "PP"]
CONF_THRESH   = 0.50

# ==================== GEOMETRY ====================
IMAGE_H = 480
IMAGE_W = 640
COMMIT_LINE_PY     = 240       # Middle of screen
COMMIT_TO_SERVO_MM = 388.0     # Measured by user
DEFAULT_MM_PER_PX  = 340.0 / IMAGE_H  # Fallback: 0.708

# ==================== TRACKING ====================
MIN_REG_POINTS = 6             # Min data points for regression
MIN_SPEED_MMPS = 8.0           # Min belt speed
MIN_R_SQUARED  = 0.70          # Min regression confidence
GRACE_FRAMES   = 25            # YOLO flicker tolerance
COOLDOWN_SEC   = 2.0
STUCK_TIMEOUT  = 5.0

# ==================== MODEL ====================
MODEL_PATH = "/home/krish/runs/detect/train_yolo11l_final/weights/best.pt"


# ===========================================================
#  CAMERA INIT (color + depth for calibration)
# ===========================================================
def init_camera():
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, IMAGE_W, IMAGE_H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, IMAGE_W, IMAGE_H, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align   = rs.align(rs.stream.color)
    depth_s = profile.get_device().first_depth_sensor()
    depth_s.set_option(rs.option.visual_preset, 3)
    intrin  = profile.get_stream(
        rs.stream.depth
    ).as_video_stream_profile().get_intrinsics()
    return pipe, align, intrin


# ===========================================================
#  AUTO-CALIBRATION: measure camera height -> exact mm/pixel
# ===========================================================
def calibrate_scale(pipe, align, intrin):
    print("\n[CAL] Measuring camera height from depth sensor...")
    depths = []
    for _ in range(30):
        frames  = pipe.wait_for_frames()
        aligned = align.process(frames)
        df      = aligned.get_depth_frame()
        if not df:
            continue
        d = df.get_distance(IMAGE_W // 2, IMAGE_H // 2)
        if 0.1 < d < 2.0:
            depths.append(d)

    if len(depths) < 5:
        print(f"[CAL] Depth failed ({len(depths)} samples).")
        print(f"[CAL] Using default: {DEFAULT_MM_PER_PX:.4f} mm/px")
        return DEFAULT_MM_PER_PX

    median_d = sorted(depths)[len(depths) // 2]

    # Deproject two points 200 pixels apart at the measured height
    cx = IMAGE_W // 2
    cy = IMAGE_H // 2
    p1 = rs.rs2_deproject_pixel_to_point(intrin, [cx, cy - 100], median_d)
    p2 = rs.rs2_deproject_pixel_to_point(intrin, [cx, cy + 100], median_d)
    dy_mm  = abs(p2[1] - p1[1]) * 1000.0
    mm_px  = dy_mm / 200.0

    h_mm   = median_d * 1000.0
    belt_mm = mm_px * IMAGE_H

    print(f"[CAL] Camera height   : {h_mm:.1f} mm")
    print(f"[CAL] Calibrated scale: {mm_px:.4f} mm/pixel")
    print(f"[CAL] Belt visible    : {belt_mm:.1f} mm across {IMAGE_H} px")
    print(f"[CAL] (Default was    : {DEFAULT_MM_PER_PX:.4f} mm/pixel)\n")
    return mm_px


# ===========================================================
#  LINEAR REGRESSION VELOCITY
# ===========================================================
def compute_velocity(track_data):
    """
    Fit a least-squares line through (time, pixel_y) data.
    Returns (slope_px_per_sec, r_squared).
    slope is negative when belt moves upward (py decreases).
    """
    n = len(track_data)
    if n < MIN_REG_POINTS:
        return 0.0, 0.0

    t_arr = np.array([d[0] for d in track_data])
    p_arr = np.array([d[1] for d in track_data])

    # Normalize time to avoid floating-point issues
    t_arr = t_arr - t_arr[0]

    if t_arr[-1] < 0.10:
        return 0.0, 0.0

    # y = slope * t + intercept
    coeffs = np.polyfit(t_arr, p_arr, 1)
    slope     = coeffs[0]
    intercept = coeffs[1]

    # R-squared (coefficient of determination)
    predicted = slope * t_arr + intercept
    ss_res = np.sum((p_arr - predicted) ** 2)
    ss_tot = np.sum((p_arr - np.mean(p_arr)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-6 else 0.0
    r2 = max(0.0, r2)

    return slope, r2


# ===========================================================
#  SERVO (software PWM bit-bang)
# ===========================================================
def move_servo(pin, angle, dur=0.5):
    pw = 0.001 + (angle / 180.0) * 0.001
    cycles = int(dur / 0.02)
    for _ in range(cycles):
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(pw)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(0.02 - pw)


def fire_sort(plastic):
    angle = BIN_ANGLES.get(plastic, 90)
    print(f"\n{'='*40}")
    print(f"  SORTING {plastic}  ->  Servo {angle} deg")
    print(f"{'='*40}")
    move_servo(BASE_SERVO_PIN, angle, 0.3)
    move_servo(STACK_SERVO_PIN, STACK_SWEEP, 0.3)
    time.sleep(0.1)
    move_servo(STACK_SERVO_PIN, STACK_HOME, 0.3)
    move_servo(BASE_SERVO_PIN, BASE_HOME, 0.3)
    print(f"  DONE - servos homed.\n")


# ===========================================================
#  MAIN
# ===========================================================
def main():
    print("=" * 50)
    print("  AI SORTING v2.3  (DEEP MATH EDITION)")
    print("=" * 50)

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(BASE_SERVO_PIN, GPIO.OUT)
    GPIO.setup(STACK_SERVO_PIN, GPIO.OUT)
    move_servo(BASE_SERVO_PIN, BASE_HOME, 0.5)
    move_servo(STACK_SERVO_PIN, STACK_HOME, 0.5)

    pipe, align, intrin = init_camera()
    print("[CAM] RealSense started (color + depth).")

    # ─── AUTO-CALIBRATE ───
    mm_px = calibrate_scale(pipe, align, intrin)

    model = YOLO(MODEL_PATH)
    print("[AI]  YOLO loaded.")

    # State
    state        = "IDLE"
    track_data   = deque(maxlen=60)  # Up to 2 sec of data
    speed_mm     = 0.0
    r_squared    = 0.0
    missed       = 0
    target_class = ""
    fire_time    = 0.0
    cooldown_end = 0.0

    print(f"\n[READY] Classes: {VALID_CLASSES}")
    print(f"[READY] Commit line at py={COMMIT_LINE_PY}")
    print(f"[READY] {COMMIT_TO_SERVO_MM}mm to servo")
    print(f"[READY] Drop a bottle on the MOVING belt!\n")

    try:
        while True:
            frames = pipe.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue
            img = np.asanyarray(cf.get_data())
            now = time.time()

            # ─── COOLDOWN ───
            if state == "COOLDOWN":
                if now > cooldown_end:
                    state     = "IDLE"
                    speed_mm  = 0.0
                    r_squared = 0.0
                    print("[SYS] Ready for next bottle.\n")
                cv2.putText(img, "COOLDOWN", (20, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (128,128,128), 2)
                cv2.imshow("AI Sorting", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ─── COMMITTED (countdown) ───
            if state == "COMMITTED":
                remain = fire_time - now
                cv2.putText(img,
                    f"Spd: {speed_mm:.1f}mm/s  R2={r_squared:.2f}",
                    (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 2)
                if remain > 0:
                    cv2.putText(img, f"IMPACT: {remain:.1f}s",
                        (20, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5,
                        (0,165,255), 3)
                else:
                    fire_sort(target_class)
                    state = "COOLDOWN"
                    cooldown_end = now + COOLDOWN_SEC
                cv2.imshow("AI Sorting", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ─── YOLO DETECTION ───
            results = model(img, verbose=False)
            det    = None
            best_c = 0.0
            for r in results:
                for box in r.boxes:
                    c   = float(box.conf[0])
                    cls = model.names[int(box.cls[0])]
                    if (cls in VALID_CLASSES
                            and c >= CONF_THRESH
                            and c > best_c):
                        best_c = c
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        det = {
                            "cls": cls, "conf": c,
                            "py": (y1 + y2) / 2.0,
                            "box": (x1, y1, x2, y2),
                        }

            # ─── IDLE -> TRACKING ───
            if state == "IDLE" and det:
                state = "TRACKING"
                track_data.clear()
                track_data.append((now, det["py"]))
                target_class = det["cls"]
                missed = 0
                print(f"[DETECT] {det['cls']} ({det['conf']:.0%}) "
                      f"py={det['py']:.0f} -> TRACKING")

            # ─── TRACKING ───
            elif state == "TRACKING":
                if det:
                    missed = 0
                    py = det["py"]
                    track_data.append((now, py))

                    # Run linear regression on ALL collected points
                    slope_px, r2 = compute_velocity(track_data)
                    speed_px  = abs(slope_px)
                    speed_mm  = speed_px * mm_px
                    r_squared = r2

                    # Distance from CURRENT position to servo
                    dist = COMMIT_TO_SERVO_MM + (py - COMMIT_LINE_PY) * mm_px

                    # Has bottle crossed the commit line? (moving UP = py decreasing)
                    crossed = (py <= COMMIT_LINE_PY)

                    if (crossed
                            and speed_mm >= MIN_SPEED_MMPS
                            and r2 >= MIN_R_SQUARED):
                        eta = dist / speed_mm if speed_mm > 0 else 999
                        if 0 < eta < 30:
                            fire_time = now + eta
                            state = "COMMITTED"
                            n = len(track_data)
                            print(
                                f"[LOCK] "
                                f"slope={slope_px:.1f}px/s  "
                                f"speed={speed_mm:.1f}mm/s  "
                                f"R2={r2:.3f}  "
                                f"pts={n}  "
                                f"dist={dist:.0f}mm  "
                                f"ETA={eta:.2f}s"
                            )

                    elif (now - track_data[0][0]) > STUCK_TIMEOUT:
                        print(
                            f"[WARN] Timeout. "
                            f"spd={speed_mm:.1f} R2={r2:.2f} -> IDLE"
                        )
                        state = "IDLE"
                        track_data.clear()
                else:
                    missed += 1
                    if missed > GRACE_FRAMES:
                        print(f"[LOST] {missed} frames -> IDLE")
                        state  = "IDLE"
                        missed = 0
                        track_data.clear()

            # ─── HUD ───
            # Yellow commit line
            cv2.line(img, (0, COMMIT_LINE_PY),
                     (IMAGE_W, COMMIT_LINE_PY), (0,255,255), 2)
            cv2.putText(img, "COMMIT",
                (IMAGE_W - 100, COMMIT_LINE_PY - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            # Detection box
            if det:
                bx = [int(v) for v in det["box"]]
                cv2.rectangle(img, (bx[0],bx[1]),
                              (bx[2],bx[3]), (0,255,0), 2)
                cv2.putText(img,
                    f"{det['cls']} {det['conf']:.0%}",
                    (bx[0], bx[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # State, speed, R2, points, scale
            sc = (0,255,0) if state == "TRACKING" else (255,255,255)
            cv2.putText(img, f"[{state}]", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2)
            cv2.putText(img, f"Speed: {speed_mm:.1f} mm/s", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(img,
                f"R2: {r_squared:.2f}  pts: {len(track_data)}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
            cv2.putText(img,
                f"Scale: {mm_px:.3f} mm/px",
                (IMAGE_W - 200, IMAGE_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            cv2.imshow("AI Sorting", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[SYS] Shutting down...")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("[SYS] Goodbye.")


if __name__ == "__main__":
    main()
