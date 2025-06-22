
# Object Tracking & Speed Estimation using YOLOv8

This repository contains code for object detection, tracking, and speed estimation of moving objects using the YOLOv8 model. It is designed to analyze object motion in videos—especially for applications like vehicle monitoring and smart traffic systems.

## 🚗 Features

- YOLOv8-based object detection  
- Object tracking across frames  
- Speed estimation using object displacement and frame rate  
- Real-time video visualization with speed overlays

## 📁 Project Structure

```
object-tracking-speed-estimation/
├── data/           # Input videos
├── output/         # Result videos with speed overlays
├── weights/        # YOLOv8 model weights
├── tracker/        # Tracking logic (e.g., centroid tracking)
├── utils/          # Helper functions (distance, speed, etc.)
├── main.py         # Main execution script
├── requirements.txt# Dependency list
└── README.md       # Project documentation
```

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/ksh99-git/object-tracking-speed-estimation.git
cd object-tracking-speed-estimation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download YOLOv8 weights:  
Pretrained weights can be downloaded from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).  
Place the file (e.g., `yolov8n.pt`) in the `weights/` folder.

## 🚀 How to Run

```bash
python main.py --video_path ./data/sample_video.mp4 --model ./weights/yolov8n.pt --show --save
```

### Optional Arguments

- `--video_path`: Path to the input video file  
- `--model`: Path to YOLOv8 weights  
- `--show`: (Optional) Show real-time output window  
- `--save`: (Optional) Save output to `output/` directory  

## 🔧 How to Run with GUI

```bash
python dcu.py
```

Make sure your sample video is placed at:

```
data/sample_video.mp4
```

To use your webcam instead, modify `dcu.py`:

```python
video_source = 0
```

## 🧠 Method Overview

1. Detect objects using YOLOv8  
2. Track objects across frames using a simple ID-assignment tracker  
3. Compute speed using:

```python
speed = distance_moved / time_elapsed
```

4. Display object ID and estimated speed on each frame

## 🎥 Sample Video

A sample input video is included in the `data/` folder:

- File: `data/sample_video.mp4`
- Description: Used for object tracking and speed estimation demonstration

## 🧰 Dependencies

- Python 3.8+  
- OpenCV  
- Ultralytics (YOLOv8)  
- NumPy  

Install using:

```bash
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Author

- GitHub: [ksh99-git](https://github.com/ksh99-git)
