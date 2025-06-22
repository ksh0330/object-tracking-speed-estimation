import sys
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
from collections import defaultdict, deque
import torch
import time

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

class VideoMonitor(QWidget):
    def __init__(self, video_source):
        super().__init__()
        self.setWindowTitle("Video Monitor")
        self.setGeometry(100, 100, 1800, 900)

        # YOLO 모델 로드
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("epoch52.pt").to(self.device)

        # Areas and colors
        self.areas = {
            "B": [(1116, 52), (775, 251), (966, 337), (1244, 87)],
            "A": [(74, 85), (361, 337), (548, 251), (200, 51)]
        }
        self.colors = {
            "A": (0, 255, 0),  # Green
            "B": (255, 0, 0)   # Blue
        }
        self.RED = (0, 0, 255)
        self.T_W, self.T_H = 160, 720

        # Video capture setup
        self.cap = cv.VideoCapture(video_source)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()

        # Video display widget
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(1280, 720)

        # Play/Pause button
        self.is_paused = False
        self.play_pause_button = QPushButton("Pause", self)
        self.play_pause_button.setGeometry(1300, 760, 100, 30)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)

        # BBox와 객체 정보 표시 토글 버튼
        self.show_bbox = True
        self.toggle_button = QPushButton("Hide BBox", self)
        self.toggle_button.setGeometry(1420, 760, 100, 30)
        self.toggle_button.clicked.connect(self.toggle_display)

        # Table widget to display object information
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(5)
        self.table_widget.setHorizontalHeaderLabels(["Area", "Track ID", "Class", "Position", "Speed (km/h)"])
        self.table_widget.setGeometry(1300, 20, 480, 300)

        # 충돌 예측 라벨
        self.collision_label = QLabel(self)
        self.collision_label.setGeometry(1300, 340, 480, 50)
        self.collision_label.setAlignment(Qt.AlignCenter)
        self.collision_label.setStyleSheet("font-size: 20px; color: red;")
        self.collision_label.setText("충돌 예측: 안전")

        # 로그 표시 라벨
        self.log_label = QLabel(self)
        self.log_label.setGeometry(1300, 400, 480, 350)
        self.log_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log_label.setStyleSheet("font-size: 12px;")
        self.log_label.setWordWrap(True)
        self.log_text = ""

        # Timer setup for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 간격으로 프레임 업데이트

        # Initialize the ViewTransformers for areas A and B
        sourceA = np.array(self.areas["A"])
        sourceB = np.array(self.areas["B"])
        target = np.array([[0, 0], [self.T_W, 0], [0, self.T_H], [self.T_W, self.T_H]])
        self.transformerA = ViewTransformer(source=sourceA, target=target)
        self.transformerB = ViewTransformer(source=sourceB, target=target)

        # Tracking data structures
        self.active_tracks = {"A": {}, "B": {}}  # 현재 영역 내에 있는 객체들
        self.exited_tracks = {"A": {}, "B": {}}  # 영역을 벗어난 객체들

    def toggle_play_pause(self):
        if self.is_paused:
            self.timer.start(30)
            self.play_pause_button.setText("Pause")
        else:
            self.timer.stop()
            self.play_pause_button.setText("Play")
        self.is_paused = not self.is_paused

    def toggle_display(self):
        self.show_bbox = not self.show_bbox
        if self.show_bbox:
            self.toggle_button.setText("Hide BBox")
        else:
            self.toggle_button.setText("Show BBox")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # YOLO 모델을 통해 객체 추적
            results = self.model.track(frame, conf=0.6, persist=True, verbose=False)
            self.process_results(frame, results)
            self.display_frame(frame)
        else:
            self.cap.release()
            self.timer.stop()

    def process_results(self, frame, results):
        current_time = time.time()
        detections = results[0]
        boxes = detections.boxes.xyxy.cpu().numpy() if detections.boxes else []
        centers = detections.boxes.xywh.cpu().numpy()[:, :2] if detections.boxes else []
        track_ids = detections.boxes.id.cpu().numpy() if detections.boxes else []
        classes = detections.boxes.cls.cpu().numpy() if detections.boxes else []

        # 현재 프레임에서 감지된 트랙 ID를 저장
        current_track_ids = {"A": set(), "B": set()}

        # 테이블 초기화
        self.table_widget.setRowCount(0)

        # 현재 프레임의 감지 결과 처리
        for box, center, track_id, cls in zip(boxes, centers, track_ids, classes):
            x1, y1, x2, y2 = map(int, box)
            x_center, y_center = center
            class_name = self.model.names[int(cls)]

            # 각 영역에 대한 처리
            for area_name, area_coords in self.areas.items():
                polygon = np.array(area_coords, np.int32)
                if cv.pointPolygonTest(polygon, (x_center, y_center), False) >= 0:
                    current_track_ids[area_name].add(track_id)

                    # 변환된 좌표 계산
                    transformer = self.transformerA if area_name == "A" else self.transformerB
                    transformed_point = transformer.transform_points(np.array([[x_center, y_center]]))[0]
                    transformed_y = transformed_point[1]

                    # 이전 프레임에서의 위치 가져오기
                    track_data = self.active_tracks[area_name].get(track_id)
                    if track_data:
                        prev_time = track_data['time']
                        prev_y = track_data['position']
                        time_diff = current_time - prev_time
                        distance = abs(transformed_y - prev_y)
                        speed = (distance / time_diff) * 3.6  # m/s 를 km/h 로 변환
                    else:
                        speed = 0.0

                    # 트랙 데이터 업데이트
                    self.active_tracks[area_name][track_id] = {
                        'position': transformed_y,
                        'time': current_time,
                        'speed': speed,
                        'class': class_name
                    }

                    # Bounding Box 및 정보 표시
                    if self.show_bbox:
                        color = self.colors[area_name]
                        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv.putText(frame, f'{area_name} ID:{track_id} {class_name} {speed:.2f}km/h',
                                   (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # 테이블에 정보 추가
                    row_position = self.table_widget.rowCount()
                    self.table_widget.insertRow(row_position)
                    self.table_widget.setItem(row_position, 0, QTableWidgetItem(area_name))
                    self.table_widget.setItem(row_position, 1, QTableWidgetItem(str(track_id)))
                    self.table_widget.setItem(row_position, 2, QTableWidgetItem(class_name))
                    self.table_widget.setItem(row_position, 3, QTableWidgetItem(f'({int(transformed_point[0])}, {int(transformed_point[1])})'))
                    self.table_widget.setItem(row_position, 4, QTableWidgetItem(f'{speed:.2f}'))

        # 영역을 벗어난 트랙 처리
        for area_name in ["A", "B"]:
            # 현재 프레임에서 감지되지 않은 트랙 ID들
            exited_track_ids = set(self.active_tracks[area_name].keys()) - current_track_ids[area_name]
            for track_id in exited_track_ids:
                track_data = self.active_tracks[area_name].pop(track_id)
                exit_time = current_time
                # 추적 유지 시간 계산 (속도에 따라)
                retention_time = self.calculate_retention_time(track_data['speed'])
                # exited_tracks에 추가
                self.exited_tracks[area_name][track_id] = {
                    'position': track_data['position'],
                    'speed': track_data['speed'],
                    'class': track_data['class'],
                    'exit_time': exit_time,
                    'retention_time': retention_time
                }
                self.log_text += f"[{time.strftime('%H:%M:%S')}] {area_name} 영역에서 ID:{track_id} 객체가 영역을 벗어남. 속도: {track_data['speed']:.2f}km/h, 추적 유지 시간: {retention_time:.2f}s\n"

        # exited_tracks에서 retention_time이 지난 트랙 제거
        for area_name in ["A", "B"]:
            to_remove = []
            for track_id, data in self.exited_tracks[area_name].items():
                if current_time - data['exit_time'] > data['retention_time']:
                    to_remove.append(track_id)
            for track_id in to_remove:
                self.exited_tracks[area_name].pop(track_id)
                self.log_text += f"[{time.strftime('%H:%M:%S')}] {area_name} 영역에서 ID:{track_id} 객체의 추적 유지 시간이 만료됨.\n"

        # 충돌 예측 수행
        self.predict_collision(current_time)

        # 로그 업데이트
        self.log_label.setText(self.log_text)

    def calculate_retention_time(self, speed):
        """
        객체의 속도에 따라 추적 유지 시간을 계산합니다.
        속도가 빠를수록 짧은 유지 시간을 가집니다.
        """
        if speed <= 0:
            return 3.0  # 기본값 3초
        max_speed = 60.0  # km/h
        min_speed = 10.0  # km/h
        max_time = 5.0    # 초
        min_time = 1.0    # 초

        speed = max(min_speed, min(speed, max_speed))  # 속도 제한
        retention_time = max_time - ((speed - min_speed) / (max_speed - min_speed)) * (max_time - min_time)
        return retention_time

    def predict_collision(self, current_time):
        """
        현재 영역 내 객체들과 영역을 벗어난 객체들의 예측 위치를 기반으로 충돌을 예측합니다.
        """
        collision_status = "안전"

        # 모든 활성 및 예측 객체들의 위치 수집
        all_objects = []

        # 활성 객체들 추가
        for area_name in ["A", "B"]:
            for track_id, data in self.active_tracks[area_name].items():
                all_objects.append({
                    'area': area_name,
                    'track_id': track_id,
                    'position': data['position'],
                    'speed': data['speed'],
                    'class': data['class'],
                    'time_since_exit': 0
                })

        # 예측 객체들 추가
        for area_name in ["A", "B"]:
            for track_id, data in self.exited_tracks[area_name].items():
                time_elapsed = current_time - data['exit_time']
                predicted_position = data['position'] + data['speed'] * (time_elapsed / 3.6)  # km/h to m/s
                all_objects.append({
                    'area': area_name,
                    'track_id': track_id,
                    'position': predicted_position,
                    'speed': data['speed'],
                    'class': data['class'],
                    'time_since_exit': time_elapsed
                })

        # 영역이 다른 객체들 간의 거리 계산
        for obj_a in all_objects:
            for obj_b in all_objects:
                if obj_a['area'] != obj_b['area']:
                    distance = abs(obj_a['position'] - obj_b['position'])
                    if distance < 50:
                        collision_status = "위험"
                        break
                    elif distance < 100 and collision_status != "위험":
                        collision_status = "주의"

        self.collision_label.setText(f"충돌 예측: {collision_status}")

    def display_frame(self, frame):
        frame = cv.resize(frame, (1280, 720))
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_source = r"D:\Straight_Data_0623\Danger\ACBC_MMD.mp4"
    window = VideoMonitor(video_source)
    window.show()
    sys.exit(app.exec_())
