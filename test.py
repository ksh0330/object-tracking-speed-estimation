import sys
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QTimer
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
        self.setGeometry(100, 100, 1800, 720)

        # YOLO 모델 로드
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("epoch52.pt").to(self.device)

        # Areas and colors
        self.areas = {
            "B": [(1116, 52), (775, 251), (966, 337), (1244, 87)],
            "A": [(74, 85), (361, 337), (548, 251), (200, 51)]
        }
        self.RED, self.BLACK, self.WHITE, self.BLUE, self.GREEN = (0, 0, 255), (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)
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
        self.play_pause_button.setGeometry(1300, 600, 100, 30)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)

        # BBox와 객체 정보 표시 토글 버튼
        self.show_bbox = True
        self.toggle_button = QPushButton("Hide BBox", self)
        self.toggle_button.setGeometry(1300, 640, 100, 30)
        self.toggle_button.clicked.connect(self.toggle_display)

        self.text_table_widget = QLabel(self)
        self.text_table_widget.setGeometry(1300, 0, 300, 20)
        self.text_table_widget.setText("모든 객체 정보")

        # Table widget to display object information
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "Confidence", "Center (X, Y)"])
        self.table_widget.setGeometry(1300, 20, 420, 100)

        self.text_a_area_table_widget = QLabel(self)
        self.text_a_area_table_widget.setGeometry(1300, 140, 300, 20)
        self.text_a_area_table_widget.setText("A 영역 내 객체 정보")

        # A영역에 있는 객체 정보 테이블 위젯 (두 번째 표)
        self.a_area_table_widget = QTableWidget(self)
        self.a_area_table_widget.setColumnCount(4)
        self.a_area_table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "Transform (X, Y)", "Speed"])
        self.a_area_table_widget.setGeometry(1300, 160, 420, 100)

        self.text_b_area_table_widget = QLabel(self)
        self.text_b_area_table_widget.setGeometry(1300, 280, 300, 20)
        self.text_b_area_table_widget.setText("B 영역 내 객체 정보")

        # B영역에 있는 객체 정보 테이블 위젯 (세 번째 표)
        self.b_area_table_widget = QTableWidget(self)
        self.b_area_table_widget.setColumnCount(4)
        self.b_area_table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "Transform (X, Y)", "Speed"])
        self.b_area_table_widget.setGeometry(1300, 300, 420, 100)

        # 충돌 예측 라벨
        self.collision_label = QLabel(self)
        self.collision_label.setGeometry(1300, 420, 500, 50)
        self.collision_label.setText("충돌 예측: 안전")

        # 텍스트 디스플레이 라벨
        self.text_label = QLabel(self)
        self.text_label.setGeometry(1300, 500, 500, 50)
        self.text_label.setText("객체 탐지 중...")

        # Timer setup for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)

        # 마지막 프레임 저장
        self.last_frame = None
        self.last_results = None

        # Initialize the ViewTransformers for areas A and B
        sourceA = np.array([[74, 85], [200, 51], [361, 337], [548, 251]])
        sourceB = np.array([[1116, 52], [1244, 87], [775, 251], [966, 337]])
        target = np.array([[0, 0], [self.T_W, 0], [0, self.T_H], [self.T_W, self.T_H]])
        self.transformerA = ViewTransformer(source=sourceA, target=target)
        self.transformerB = ViewTransformer(source=sourceB, target=target)

        # Initialize variables for drawing
        self.T_FPS = 5
        self.PREDICT = 12
        self.track_history = defaultdict(lambda: deque(maxlen=self.T_FPS))
        self.trackmA_history = defaultdict(lambda: deque(maxlen=self.T_FPS))
        self.trackmB_history = defaultdict(lambda: deque(maxlen=self.T_FPS))
        self.predictA_history = defaultdict(lambda: deque(maxlen=self.PREDICT))
        self.predictB_history = defaultdict(lambda: deque(maxlen=self.PREDICT))

        self.A_slope, self.B_slope = 0, 0
        self.A_v_list, self.B_v_list = [], []

        # 충돌 예측 정보 저장
        self.last_a_position = None
        self.last_b_position = None
        self.last_a_speed = None
        self.last_b_speed = None

        self.current_a_position = None
        self.current_b_position = None
        self.current_a_speed = None
        self.current_b_speed = None

        # 객체의 출구 시간을 기록하기 위한 딕셔너리
        self.exit_times = {}

    def calculate_time_to_track(self, speed):
        max_speed = 23  # 가장 빠른 속도
        min_speed = 7   # 가장 느린 속도
        min_time = 1    # 가장 빠를 때 추적 시간 (1초)
        max_time = 5    # 가장 느릴 때 추적 시간 (5초)

        if speed >= max_speed:
            return min_time
        elif speed <= min_speed:
            return max_time
        else:
            ratio = (max_speed - speed) / (max_speed - min_speed)
            return min_time + (max_time - min_time) * ratio

    def toggle_play_pause(self):
        if self.is_paused:
            self.timer.start(0)
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

        if self.last_frame is not None and self.last_results is not None:
            self.update_display(self.last_frame, self.last_results)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # YOLO 모델을 통해 객체 추적
            results = self.model.track(frame, conf=0.6, imgsz=(736, 1280), persist=True, verbose=False)
            self.last_frame = frame.copy()
            self.last_results = results

            # 화면 업데이트
            self.update_display(frame, results)
        else:
            self.cap.release()

    def predict_collision(self, current_pos, last_pos, speed, direction):
        future_position = current_pos + (current_pos - last_pos)
        displacement = future_position - current_pos
        displacement_direction = np.sign(displacement)
        future_position_adjusted = future_position + (speed * displacement_direction * direction)
        return future_position_adjusted

    def draw_tracking_path(self, frame, track_history, color=(0, 255, 0)):
        for track_id, points in track_history.items():
            if len(points) > 1:
                points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv.polylines(frame, [points_array], isClosed=False, color=color, thickness=2)

    def update_display(self, frame, results):
        self.table_widget.setRowCount(0)
        self.a_area_table_widget.setRowCount(0)
        self.b_area_table_widget.setRowCount(0)

        text_to_display = ""
        collision_status = "안전"
        current_time = time.time()
        time_to_track_a = None
        time_to_track_b = None

        # 각 객체에 대해 BBox 그리기
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            boxes_C = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            names = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, box_C, track_id, name, conf in zip(boxes, boxes_C, track_ids, names, confs):
                x1, y1, x2, y2 = map(int, box)
                x, y, w, h = map(int, box_C)
                cls_name = self.model.names[int(name)]

                track = self.track_history[track_id]
                track.append((x, y))

                transformed_pointA = self.transformerA.transform_points(np.array([[x, y]]))
                transformed_pointB = self.transformerB.transform_points(np.array([[x, y]]))

                transformed_textA = f'({int(transformed_pointA[0][0])}, {int(transformed_pointA[0][1])})'
                transformed_textB = f'({int(transformed_pointB[0][0])}, {int(transformed_pointB[0][1])})'

                if self.show_bbox:
                    for area_name, area_coords in self.areas.items():
                        if cv.pointPolygonTest(np.array(area_coords, np.int32), (x, y), False) >= 0:
                            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv.putText(frame, f'ID: {int(track_id)} {cls_name}',
                                       (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.GREEN, 2)

                            self.draw_tracking_path(frame, self.track_history, color=self.GREEN)

                            if area_name == "A":
                                trackma = self.trackmA_history[track_id]
                                trackma.append(transformed_pointA)
                                self.A_v_list.append(int(transformed_pointA[0][1]))
                                self.A_slope = round(abs(self.A_v_list[-1] - self.A_v_list[0]) / len(self.A_v_list), 2)
                                av = self.A_slope
                                av1 = round((av * 0.0025 * 30), 2)
                                av2 = round((av1 * 3.6), 2)
                                av3 = round((av2 * 7.5), 2)

                                self.current_a_position = transformed_pointA[0][1]
                                self.current_a_speed = av3

                                row_position = self.a_area_table_widget.rowCount()
                                self.a_area_table_widget.insertRow(row_position)
                                self.a_area_table_widget.setItem(row_position, 0, QTableWidgetItem(str(int(track_id))))
                                self.a_area_table_widget.setItem(row_position, 1, QTableWidgetItem(cls_name))
                                self.a_area_table_widget.setItem(row_position, 2, QTableWidgetItem(transformed_textA))
                                self.a_area_table_widget.setItem(row_position, 3, QTableWidgetItem(str(av3)))
                                text_to_display += f"A 영역: ID: {int(track_id)}, Name: {cls_name}, 변환된 좌표: {transformed_textA}, 속도: {av3}\n"

                                time_to_track_a = self.calculate_time_to_track(av3)
                                self.exit_times[track_id] = current_time + time_to_track_a
                                print(
                                    f"Track ID {track_id} will be excluded from collision prediction after {time_to_track_a} seconds.")

                                if time_to_track_a is not None:
                                    self.exit_times[(track_id, 'A')] = current_time + time_to_track_a
                                    print(
                                        f"Track ID {track_id} will be excluded from collision prediction after {time_to_track_a} seconds.")



                        elif area_name == "B":
                                trackmb = self.trackmB_history[track_id]
                                trackmb.append(transformed_pointB)
                                self.B_v_list.append(int(transformed_pointB[0][1]))
                                self.B_slope = round(abs(self.B_v_list[-1] - self.B_v_list[0]) / len(self.B_v_list), 2)
                                bv = self.B_slope
                                bv1 = round((bv * 0.0025 * 30), 2)
                                bv2 = round((bv1 * 3.6), 2)
                                bv3 = round((bv2 * 7.5), 2)

                                self.current_b_position = transformed_pointB[0][1]
                                self.current_b_speed = bv3

                                row_position = self.b_area_table_widget.rowCount()
                                self.b_area_table_widget.insertRow(row_position)
                                self.b_area_table_widget.setItem(row_position, 0, QTableWidgetItem(str(int(track_id))))
                                self.b_area_table_widget.setItem(row_position, 1, QTableWidgetItem(cls_name))
                                self.b_area_table_widget.setItem(row_position, 2, QTableWidgetItem(transformed_textB))
                                self.b_area_table_widget.setItem(row_position, 3, QTableWidgetItem(str(bv3)))
                                text_to_display += f"B 영역: ID: {int(track_id)}, Name: {cls_name}, 변환된 좌표: {transformed_textB}, 속도: {bv3}\n"

                                time_to_track_b = self.calculate_time_to_track(bv3)
                                self.exit_times[track_id] = current_time + time_to_track_b
                                print(
                                    f"Track ID {track_id} will be excluded from collision prediction after {time_to_track_b} seconds.")

                                if time_to_track_b is not None:
                                    self.exit_times[(track_id, 'B')] = current_time + time_to_track_b
                                    print(
                                        f"Track ID {track_id} will be excluded from collision prediction after {time_to_track_b} seconds.")

                        if (track_id, 'A') in self.exit_times and current_time > self.exit_times[(track_id, 'A')]:
                            del self.exit_times[(track_id, 'A')]
                            print(f"Track ID {track_id} in A has been removed from collision prediction.")
                        if (track_id, 'B') in self.exit_times and current_time > self.exit_times[(track_id, 'B')]:
                            del self.exit_times[(track_id, 'B')]
                            print(f"Track ID {track_id} in B has been removed from collision prediction.")

                        if self.current_a_position is not None and self.current_b_position is not None:
                            if self.last_a_position is not None and self.last_b_position is not None:
                                a_future_position = self.predict_collision(
                                    self.current_a_position, self.last_a_position, self.current_a_speed, 1)
                                b_future_position = self.predict_collision(
                                    self.current_b_position, self.last_b_position, self.current_b_speed, -1)

                                distance = np.linalg.norm(a_future_position - b_future_position)
                                if distance < 150:
                                    collision_status = "위험"
                                elif distance < 300:
                                    collision_status = "주의"
                                else:
                                    collision_status = "안전"

                        self.collision_label.setText(f"충돌 예측: {collision_status}")
                        self.last_a_position = self.current_a_position
                        self.last_b_position = self.current_b_position
                        self.last_a_speed = self.current_a_speed
                        self.last_b_speed = self.current_b_speed

                self.text_label.setText(text_to_display or "객체 탐지 중...")

                row_position = self.table_widget.rowCount()
                self.table_widget.insertRow(row_position)
                self.table_widget.setItem(row_position, 0, QTableWidgetItem(str(int(track_id))))
                self.table_widget.setItem(row_position, 1, QTableWidgetItem(cls_name))
                self.table_widget.setItem(row_position, 2, QTableWidgetItem(f'{conf:.2f}'))
                self.table_widget.setItem(row_position, 3, QTableWidgetItem(f'({x}, {y})'))

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
    video_source = r"D:\Straight_Data_0623\Safe\ACBC_FM1.mp4"
    window = VideoMonitor(video_source)
    window.show()
    sys.exit(app.exec_())
