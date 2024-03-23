import sys
import time

from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, QThread, pyqtSignal, pyqtSlot, QRect

import numpy as np
import video_processing_2 as vp

sys.setrecursionlimit(10**6)

from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QProgressBar,
    QPushButton, QFileDialog, QWidget, QSlider, QStackedWidget, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap
import cv2
import time
import sys
import pandas as pd
from xarray import DataArray
import xarray as xr
# import VSC_Minian_demo_videos as VSC

# Handles communication to and from subprocesses such as Play_video and Save_changes_2_xarray
# and communication to and from user inputs from MainWindow
class Processing_HUB(QObject):
    updateFrame = pyqtSignal(QImage)
    updateProgressBar = pyqtSignal(int)
    
    def __init__(self):
        super(Processing_HUB, self).__init__()
        self.data_array = None
        self.play_video_instance = None
        self.pause=False
 
    def handle_update_frame(self, qimage):
        if qimage is not None:
            self.updateFrame.emit(qimage)

    def play(self):
            if self.data_array is not None and len(self.data_array) > 0:
                self.play_video_instance = Play_video(self.data_array)
                self.play_video_instance.updateFrame.connect(self.handle_update_frame)
                self.play_video_instance.start()

    def load_xarray_perframe(self, video_path):
        self.data_array = xr.open_dataset(video_path)

    def set_file_path(self, file_path):
        self.load_xarray_perframe(file_path)
    
    @pyqtSlot()
    def resume(self): 
        if self.play_video_instance:
            self.play_video_instance.resume_thread()
            self.pause=False

    @pyqtSlot()
    def stop_thread_play(self):
        if self.play_video_instance:
            self.play_video_instance.pause_thread()
            self.pause=True


class Play_video(QThread):
    updateFrame = pyqtSignal(QImage)

    def __init__(self, data_array):
        super(Play_video, self).__init__()
        self.data_array = data_array
        self.frame_rate = 5
        self.stop_thread = False
        self.frame_index = 0  # Initialize frame index
        self.convert_2_contours = False

    def run(self):
        self.get_video()

    def get_video(self):
        print('Playing')
        self.stop_thread = False
        time_frame = 1 / self.frame_rate
        if self.data_array is not None:
            while not self.stop_thread and self.frame_index < len(self.data_array["frame"]):
                for i in range(self.frame_index, len(self.data_array["frame"])):
                    if not self.stop_thread:
                        img = self.data_array["__xarray_dataarray_variable__"][i].values
                        if img is not None:
                            height, width, _ = img.shape  # Get height and width from frame
                            q_img = self.array_2_qimg(img)
                            self.updateFrame.emit(q_img)
                            self.frame_index += 1  # Increment frame index
                            time.sleep(time_frame)
                        else:
                            print('Failed to get frame')
                            continue

    def pause_thread(self):
        self.stop_thread = True

    def resume_thread(self):
        self.stop_thread = False

    def array_2_qimg(self, frame):
        if len(frame.shape) == 3:
            height, width, _ = frame.shape
            if frame.dtype != np.uint8:  # Ensure data type is uint8
                frame = frame.astype(np.uint8)
            return QImage(frame.data, width, height, frame.strides[0], QImage.Format_RGB888)
        elif len(frame.shape) == 2:
            height, width = frame.shape
            if frame.dtype != np.uint8:  # Ensure data type is uint8
                frame = frame.astype(np.uint8)
            return QImage(frame.data, width, height, frame.strides[0], QImage.Format_Grayscale8)
        else:
            print('no image conversion')
            return None


class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("xarray_player")
        self.setGeometry(0, 0, 800, 500)
        outerLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        Button_layout = QHBoxLayout()

        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        self.upload_button = QPushButton("Upload Video", self)
        self.upload_button.clicked.connect(self.open_file_dialog)

        self.button_play = QPushButton("Start")
        self.button_play.clicked.connect(self.start_thread)
        self.button_stop = QPushButton("Pause")
        self.button_stop.clicked.connect(self.stop_thread)

        topLayout.addWidget(self.label) 
        topLayout.addWidget(self.upload_button)

        Button_layout.addWidget(self.button_play,2)
        Button_layout.addWidget(self.button_stop,2)

        outerLayout.addLayout(topLayout)
        outerLayout.addLayout(Button_layout)
        self.setLayout(outerLayout)

        self.thread= Processing_HUB()
        self.thread.updateFrame.connect(lambda image: self.displayFrame(image))

    @pyqtSlot()
    def start_thread(self):
        self.thread.resume()
        self.thread.play()
        self.upload_button.setVisible(False)

    @pyqtSlot(QImage)
    def displayFrame(self, Image):
        self.label.setPixmap(QPixmap.fromImage(Image))

    @pyqtSlot()
    def stop_thread(self):
        self.thread.stop_thread_play()

    @pyqtSlot()
    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.nc)")
        if file_path:
            self.thread.set_file_path(file_path)
            self.thread.play()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())