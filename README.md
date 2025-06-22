
# Object Tracking & Speed Estimation using YOLOv8

This repository contains a simplified implementation for object detection, tracking, and speed estimation using the YOLOv8 model. It is designed to monitor object movement (e.g., vehicles) in CCTV or recorded video.

## 🚗 Features

- YOLOv8-based object detection  
- Real-time object tracking  
- Speed estimation using object displacement between frames  
- PyQt-based video monitoring interface

## 📁 Project Structure

```
object-tracking-speed-estimation/
├── data/             # Input video files
├── dcu.py            # Main GUI-based monitoring script
├── yolov8s.pt        # YOLOv8 model weights
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
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

3. Download or verify `yolov8s.pt` exists in the root directory.

## 🚀 Run the Application

```bash
python dcu.py
```

- Default video source: `data/sample_video.mp4`
- To use webcam input, modify `dcu.py` as:

```python
video_source = 0
```

## 🧠 Method Overview

- YOLOv8 detects objects frame-by-frame
- Tracks objects using centroid matching
- Estimates speed using simple displacement/time logic

## 🎥 Sample Video

Sample test video is located in the `data/` directory.

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Author

- GitHub: [ksh99-git](https://github.com/ksh99-git)
