import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QSlider, QFileDialog


class VideoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cap = None

        self.initUI()

        self.video_path = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None  # Store the current frame

        # Enable drag and drop for the main window
        self.setAcceptDrops(True)

        # Variables to store adjustment values
        self.brightness_value = 50
        self.contrast_value = 50
        self.saturation_value = 50

    def initUI(self):
        # Main Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Enable drag and drop for the central widget
        self.central_widget.setAcceptDrops(True)

        # Layout
        self.layout = QVBoxLayout(self.central_widget)

        # Video Display Placeholder
        self.video_display_label = QLabel("Video Display Area", self)
        self.layout.addWidget(self.video_display_label)

        # Brightness Label and Slider
        self.brightness_label = QLabel("Brightness", self)
        self.layout.addWidget(self.brightness_label)
        self.brightness_slider = QSlider(Qt.Horizontal, self)
        self.brightness_slider.valueChanged[int].connect(self.on_brightness_change)
        self.layout.addWidget(self.brightness_slider)

        # Contrast Label and Slider
        self.contrast_label = QLabel("Contrast", self)
        self.layout.addWidget(self.contrast_label)
        self.contrast_slider = QSlider(Qt.Horizontal, self)
        self.contrast_slider.valueChanged[int].connect(self.on_contrast_change)
        self.layout.addWidget(self.contrast_slider)

        # Saturation Label and Slider
        self.saturation_label = QLabel("Saturation", self)
        self.layout.addWidget(self.saturation_label)
        self.saturation_slider = QSlider(Qt.Horizontal, self)
        self.saturation_slider.valueChanged[int].connect(self.on_saturation_change)
        self.layout.addWidget(self.saturation_slider)

        # Set window properties
        self.setWindowTitle("Video Editor")
        self.setGeometry(300, 300, 800, 600)

        # Set ranges for sliders
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(50)

        self.contrast_slider.setRange(0, 100)
        self.contrast_slider.setValue(50)

        self.saturation_slider.setRange(0, 100)
        self.saturation_slider.setValue(50)

        # Enable drag and drop for the main widget
        self.central_widget.setAcceptDrops(True)
    
    def dragEnterEvent(self, event):
        print("Drag event triggered")  # Debug print
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    # Override dropEvent
    def dropEvent(self, event):
        print("Drop event triggered")  # Debug print
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            print(f"Dropped file path: {file_path}")  # Debug print

            if file_path.lower().endswith('.avi'):
                print("Loading AVI video...")  # Debug print
                self.load_video(file_path)
            else:
                print("The file is not an AVI video.")  # Debug print
        event.acceptProposedAction()



    def on_brightness_change(self, value):
        self.brightness_value = value
        self.update_frame()

    def on_contrast_change(self, value):
        self.contrast_value = value
        self.update_frame()

    def on_saturation_change(self, value):
        self.saturation_value = value
        self.update_frame()

    def adjust_brightness(self, frame, value):
        # Convert the value to a scale (you might need to adjust the scale factor)
        brightness_factor = (value - 50) / 25.0
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness_factor)
        return frame

    def adjust_contrast(self, frame, value):
        # Convert the value to a scale (you might need to adjust the scale factor)
        contrast_factor = 1 + (value - 50) / 50.0
        frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=0)
        return frame

    def adjust_saturation(self, frame, value):
        # Convert frame to HSV (hue, saturation, value)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Convert the value to a scale (you might need to adjust the scale factor)
        saturation_factor = (value - 50) / 50.0
        s = cv2.convertScaleAbs(s, alpha=saturation_factor, beta=0)

        # Merge back the channels and convert to BGR
        hsv = cv2.merge([h, s, v])
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame


    def load_video(self, file_path=None):
        # Release existing video capture if it exists
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if not file_path:
            self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi)")
        else:
            self.video_path = file_path

        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.timer.start(30)

    def update_frame(self):
        if self.cap is None:
            print("self.cap is None")
        elif not self.cap.isOpened():
            print("self.cap is not opened")
        else:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                processed_frame = self.process_frame(frame)
                self.display_frame(processed_frame)
            else:
                print("Failed to read frame")


    def process_frame(self, frame):
        # Retrieve the current values from the sliders
        brightness_value = self.brightness_slider.value()
        contrast_value = self.contrast_slider.value()
        saturation_value = self.saturation_slider.value()

        # Apply brightness adjustment
        frame = self.adjust_brightness(frame, brightness_value)

        # Apply contrast adjustment
        frame = self.adjust_contrast(frame, contrast_value)

        # Apply saturation adjustment
        frame = self.adjust_saturation(frame, saturation_value)

        return frame

    def display_frame(self, frame):
        # Convert frame to format suitable for PyQt and display
        qformat = QImage.Format_Indexed8 if len(frame.shape) == 2 else QImage.Format_RGB888
        out_image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        self.video_display_label.setPixmap(QPixmap.fromImage(out_image))

def main():
    app = QApplication(sys.argv)
    ex = VideoEditor()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
