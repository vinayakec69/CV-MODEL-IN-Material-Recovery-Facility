# ♻️ SmartSeg: Edge-AI Based Plastic Waste Sorting System

![NVIDIA Jetson](https://img.shields.io/badge/Hardware-NVIDIA_Jetson_Orin_Nano-76B900?style=flat-square&logo=nvidia)
![Intel RealSense](https://img.shields.io/badge/Camera-Intel_RealSense_Depth-0071C5?style=flat-square&logo=intel)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8_Classification-00FFFF?style=flat-square)
![Python](https://img.shields.io/badge/Language-Python_3.10-3776AB?style=flat-square&logo=python)

SmartSeg is a low-cost, high-performance Edge-AI solution designed to automate the segregation of plastic waste on high-speed conveyor belts. By combining the computational efficiency of the YOLOv8 image classification architecture with the NVIDIA Jetson Orin Nano, the system performs real-time material classification (PET, HDPE, PP, LDPE, MISC) at 31+ FPS.

This project was developed at the **Emphasis Lab**.

## ✨ Key Features
- **Zero-Domain-Gap Dataset:** Trained on a highly curated, custom dataset of 3,600 images captured directly on the physical conveyor belt to eliminate environmental noise and overfitting.
- **Depth-Gating Algorithm:** Utilizes the Intel RealSense depth stream to trigger the inference pipeline *only* when an object is physically present on the belt, saving massive amounts of GPU memory.
- **Multi-Frame Majority Voting:** Uses a rolling buffer supermajority rule (7 out of 10 frames) to completely eliminate single-frame hallucinations caused by motion blur or lighting anomalies.
- **Hardware Optimized:** Bypasses computationally expensive RGB-Depth alignment and runs on a stripped-down headless OS mode (`multi-user.target`) to maximize the Jetson Orin Nano's unified memory.

## 🛠 Hardware Requirements
* NVIDIA Jetson Orin Nano (8GB)
* Intel RealSense Depth Camera (e.g., D435i / D415)
* Physical Conveyor Belt Setup

## 💻 Software & Dependencies
* JetPack 5.x / 6.x (Ubuntu 20.04/22.04)
* Python 3.10+
* Ultralytics (YOLOv8)
* PyRealSense2
* OpenCV (`cv2`)
* PyTorch (Jetson optimized build)

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SmartSeg.git
   cd SmartSeg
   ```

2. **Set up the virtual environment**
   ```bash
   python3 -m venv smartseg_env
   source smartseg_env/bin/activate
   pip install -r requirements.txt
   ```

3. **Maximize GPU Memory (Recommended for Jetson)**
   To prevent `nvlm success internal assert failed` memory crashes, disable the Ubuntu GUI before running:
   ```bash
   sudo systemctl isolate multi-user.target
   ```

## 🎮 Usage

Run the real-time inference script. Ensure your trained `best.pt` YOLOv8 weights file is in the root directory.

```bash
python3 smartseg_lite.py
```
*Press `q` to quit the camera stream.*

## 🔮 Future Scope
- **Physical Actuation:** Interfacing the Jetson Orin Nano with an ESP32 microcontroller via serial communication to trigger pneumatic actuators for physical bin sorting.
- **Sensor Fusion:** Integrating a Near-Infrared (NIR) spectrometer alongside the RGB-Depth camera to accurately resolve the chemical composition of visually ambiguous plastics (like LDPE films and MISC).
- **Active Learning:** Semi-autonomously saving low-confidence images directly to local storage during operation for continuous model retraining.

## 🤝 Acknowledgements
* Emphasis Lab at Sahyadri College of Engineering and Management (SCEM), Adyar.
* Ultralytics for the YOLOv8 architecture.
