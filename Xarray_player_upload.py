import sys
import time

import cv2
from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QLabel
from xarray.core.dataarray import DataArray
import numpy as np

class Threading(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path = 'c:\\Users\\heather\\Desktop\\GUI\\Sample_avi_clip.avi'
        self.frame_rate = 30  # Frames per second
        self.data_array = None
        self.stats = None
        self.stop_thread = False

    def run(self):
        self.load_avi_perframe(self.video_path)
        self.frame_array_2_xarray()
        self.get_video()

    def load_avi_perframe(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_array = np.empty((frame_number, height, width), dtype=np.float64)
        adj = (1 / 255)  # adjusts to [0 1] scale
        for i in range(frame_number):
            ret, frame = cap.read()
            if ret:
                frame_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_array[i] = frame_conv * adj
            else:
                break
        self.frame_array = frame_array
        self.stats = [frame_number, height, width]

    def frame_array_2_xarray(self):
        self.data_array = DataArray(
            self.frame_array,
            dims=["frame", "height", "width"],
            coords={
                "frame": np.arange(self.stats[0]),
                "height": np.arange(self.stats[1]),
                "width": np.arange(self.stats[2]),
            },
        )

    def get_video(self):
        if self.data_array is not None:
            for i in range(len(self.data_array)):
                img = self.data_array[i].values
                img = np.uint8(img * 255)  # Convert back to uint8 for display
                height, width = img.shape
                q_img = QImage(img, width, height, width, QImage.Format_Grayscale8)
                self.updateFrame.emit(q_img)
                time.sleep(1 / self.frame_rate)

    def stop(self):
        self.stop_thread = True

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("xarray_player")
        self.setGeometry(0, 0, 800, 500)

        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)
        
        self.button1 = QPushButton("Start")
        self.button1.clicked.connect(self.start_thread)
        self.button_stop = QPushButton("Stop")
        self.button_stop.clicked.connect(self.stop_thread)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button1)
        layout.addWidget(self.button_stop)
        self.setLayout(layout)

        self.thread = Threading(self)
        self.thread.updateFrame.connect(self.displayFrame)

    @Slot()
    def start_thread(self):
        self.thread.start()

    @Slot(QImage)
    def displayFrame(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    @Slot()
    def stop_thread(self):
        self.thread.stop()        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())