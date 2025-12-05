# Computer Vision: Intelligent People Counter

This project implements an advanced computer vision system for monitoring and analyzing pedestrian movement within a defined Region of Interest (ROI) captured by a static camera.

## Project goal 

### Core requirements 
* **Unique Person Counting:** Accurately count non-repeating individuals who cross a defined counting line/region.

### Implementation level 
* **Low-Level Implementation:** Achieved through custom tracking logic (managing track history and IDs) and utilization of optimized, low-level linear algebra libraries (BLAS/LAPACK) via **NumPy** and **PyTorch** for feature distance calculation and model inference.

---

## Application features (some are not yet fully implemented)

The application includes an improved tracking system and lays the foundation for advanced statistical extensions ($T_b$).

### 1. Advanced tracking & counting

* **YOLOv8 Detection:** Uses the powerful YOLOv8 model for real-time person detection.
* **Unique Pass Counting:** Implements **pass-through logic** based on track history to count a person only once when they traverse the defined central region.
* **Occlusion Resistance:** Achieved through stable object tracking combined with custom history management to minimize ID swapping during brief occlusions.

### 2. Statistical analysis

The application displays and calculates the following statistics in real-time, focusing on tracks within the central region:

* **Average Speed (px/s):** Calculates the average speed of movement based on positional changes between frames and the video's FPS.
* **Average Time in Region (s):** Calculates the average duration (in seconds) that counted individuals spend inside the central region.
* **Clothing Color Estimation:** Estimates the dominant BGR color of the upper body (clothing) for visualization, demonstrating semantic segmentation capability.
