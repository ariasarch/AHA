import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import cv2

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.current_frame = None  # Store the current frame
        self.brightness_value = 50  # Initial brightness value
        self.contrast_value = 50  # Initial contrast value
        self.saturation_value = 50  # Initial saturation value
        self.playing = False
        self.speed_factor = 2.0

        self.layout = QVBoxLayout()

        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        self.select_button = QPushButton('Select Video File', self)
        self.select_button.clicked.connect(self.open_video_file)
        self.layout.addWidget(self.select_button)

        self.play_button = QPushButton('Play', self)
        self.play_button.clicked.connect(self.play)
        self.layout.addWidget(self.play_button)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop)
        self.layout.addWidget(self.stop_button)

        self.speed_label = QLabel("Speed:", self)
        self.layout.addWidget(self.speed_label)

        self.speed_slider = QSlider(Qt.Horizontal, self)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.change_speed)
        self.layout.addWidget(self.speed_slider)

        self.save_button = QPushButton('Save Video', self)
        self.save_button.clicked.connect(self.save_video)
        self.layout.addWidget(self.save_button)

        # Brightness Label and Slider
        self.brightness_label = QLabel("Brightness", self)
        self.layout.addWidget(self.brightness_label)
        self.brightness_slider = QSlider(Qt.Horizontal, self)
        self.brightness_slider.valueChanged.connect(self.on_brightness_change)
        self.layout.addWidget(self.brightness_slider)

        # Contrast Label and Slider
        self.contrast_label = QLabel("Contrast", self)
        self.layout.addWidget(self.contrast_label)
        self.contrast_slider = QSlider(Qt.Horizontal, self)
        self.contrast_slider.valueChanged.connect(self.on_contrast_change)
        self.layout.addWidget(self.contrast_slider)

        # Saturation Label and Slider
        self.saturation_label = QLabel("Saturation", self)
        self.layout.addWidget(self.saturation_label)
        self.saturation_slider = QSlider(Qt.Horizontal, self)
        self.saturation_slider.valueChanged.connect(self.on_saturation_change)
        self.layout.addWidget(self.saturation_slider)

        self.setLayout(self.layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Set window properties
        self.setWindowTitle("Video Editor")
        self.setGeometry(500, 100, 1000, 800)

        # Set ranges for sliders
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(self.brightness_value)

        self.contrast_slider.setRange(0, 100)
        self.contrast_slider.setValue(self.contrast_value)

        self.saturation_slider.setRange(0, 100)
        self.saturation_slider.setValue(self.saturation_value)

    def open_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Video File', '', 'Video files (*.mp4 *.avi *.mkv)')
        if file_path:
            self.play_video(file_path)

    def on_brightness_change(self, value):
        self.brightness_value = value

    def on_contrast_change(self, value):
        self.contrast_value = value

    def on_saturation_change(self, value):
        self.saturation_value = value

    def adjust_brightness(self, frame, value):
        brightness_factor = (value - 50) / 25.0
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness_factor)
        return frame

    def adjust_contrast(self, frame, value):
        contrast_factor = 1 + (value - 50) / 50.0
        frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=0)
        return frame

    def adjust_saturation(self, frame, value):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        saturation_factor = (value - 50) / 50.0
        s = cv2.convertScaleAbs(s, alpha=saturation_factor, beta=0)

        hsv = cv2.merge([h, s, v])
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame

    def play(self):
        if self.cap is None:
            return

        if not self.playing:
            self.timer.start(500//int(self.speed_factor))
            self.playing = True

    def stop(self):
        self.timer.stop()
        self.cap.release()
        self.playing = False

    def change_speed(self, value):
        self.speed_factor = value / 100.0

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame")
            return

        self.current_frame = frame
        processed_frame = self.process_frame(frame)
        self.display_frame(processed_frame)

    def process_frame(self, frame):
        # Apply brightness adjustment
        frame = self.adjust_brightness(frame, self.brightness_value)

        # Apply contrast adjustment
        frame = self.adjust_contrast(frame, self.contrast_value)

        # Apply saturation adjustment
        frame = self.adjust_saturation(frame, self.saturation_value)

        return frame

    def display_frame(self, frame):
        # Convert the frame to RGB format and set it as the label's pixmap
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)

    def play_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if self.cap.isOpened():
            self.timer.start(30)
    def save_video(self):
        if self.current_frame is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Video', '', 'Video files (*.avi)')
        if file_path:
            height, width, _ = self.current_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(file_path, fourcc, 30.0, (width, height))

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                out.write(frame)

            out.release()
            self.cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())