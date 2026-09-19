# KRUX: Architecture Diagrams

Below are the Mermaid architecture diagrams detailing the hardware, software, and mechanical logic of the centralized MRF Conveyor system.

### Figure 5: Complete Conveyor Prototype (Architecture Block Diagram)

```mermaid
graph TD
    %% IEEE Academic Grayscale Theme
    classDef default fill:#ffffff,stroke:#000000,stroke-width:1px,color:#000000,shape:rect
    classDef dashed_box fill:none,stroke:#000000,stroke-width:1px,stroke-dasharray: 5 5,color:#000000
    
    subgraph System[MRF Prototype Edge-Compute Architecture]
        direction TB
        A[Intel RealSense D435 Sensor] -->|Color & IR Depth| B[NVIDIA Jetson AGX Orin]
        C[Continuous Conveyor Belt] -->|Material Flow| A
        B -->|PWM Signal| D[Base Pan Servo]
        B -->|PWM Signal| E[Stack Sweep Servo]
        D -->|Target Alignment| F[Sorting Bins]
        E -->|Mechanical Actuation| F
    end
    
    class System dashed_box
```

### Figure 6: YOLO Detection + Tracking Screen (Software Flowchart)

```mermaid
flowchart LR
    %% IEEE Academic Grayscale Theme
    classDef default fill:#ffffff,stroke:#000000,stroke-width:1px,color:#000000
    
    A[Raw Video Frame] --> B{YOLOv11 Inference}
    B -->|Confidence < 0.50| C[Discard Frame]
    B -->|Confidence >= 0.50| D[Extract Bounding Box]
    D --> E{Class Filter}
    E -->|PET / HDPE / PP| F[Extract Y-Coordinate]
    E -->|Other| C
    F --> G[(Tracking Buffer)]
```

### Figure 7: ETA/Velocity Output (Mathematical State Diagram)

```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> TRACKING : Object Detected
    TRACKING --> TRACKING : Buffer (t, y) Data Points
    TRACKING --> COMMITTED : Cross Commit Line (py = 240)
    
    state COMMITTED {
        direction TB
        Regression: Least-Squares Linear Regression
        Validation: Verify R² >= 0.70
        ETA_Calc: Calculate ETA = d / v
        Wait: Asynchronous Timer Thread
        
        Regression --> Validation
        Validation --> ETA_Calc
        ETA_Calc --> Wait
    }
    
    COMMITTED --> COOLDOWN : ETA = 0.0s (Actuation)
    COOLDOWN --> IDLE : Post-Actuation Delay
```

### Figure 8: Servo Sorting Action (Sequence Diagram)

```mermaid
sequenceDiagram
    autonumber
    participant T as System Timer
    participant IO as GPIO Controller
    participant Base as Base Servo
    participant Stack as Stack Servo
    
    T->>IO: Countdown Expires (T-0.0s)
    Note right of IO: Target Assignment: PET (45°)
    IO->>Base: Transmit PWM (45°)
    Base-->>IO: Alignment Complete
    IO->>Stack: Transmit PWM (Sweep to 130°)
    Note right of Stack: Physical Impact
    IO->>Stack: Transmit PWM (Return 0°)
    Stack-->>IO: Retraction Complete
    IO->>Base: Transmit PWM (Return 90°)
```

### Figure 9: Performance Graph (Data Chart)

```mermaid
xychart-beta
    title "System Sorting Accuracy (%) vs. Polymer Class"
    x-axis ["PET (20mm/s)", "HDPE (20mm/s)", "PP (20mm/s)", "PET (>40mm/s)"]
    y-axis "Accuracy (%)" 0 --> 100
    bar [99, 98, 97, 89]
```
