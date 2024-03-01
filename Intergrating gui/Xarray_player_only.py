import sys
import time

import cv2
from PySide6.QtCore import QThread, Signal, Slot, Qt, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QLabel, QFileDialog, QSlider, QStackedWidget, QWidget
from xarray.core.dataarray import DataArray
import numpy as np
import video_processing as vp
import netCDF4 as nc
import xarray as xr

sys.setrecursionlimit(10**6)

class Threading(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):

        super().__init__(parent)
        self.frame_rate = 30  # Frames per second
        self.data_array = None
        self.stats = None
        self.stop_thread = False
        self.frame_index = 0  # Presets starting frame to index 0
        self.button_index= 0 
        self.slider_value = None
        self.current_function_index = 0
        self.limit=10**6
        self.init_slider_val=5

    def run(self):
        sys.setrecursionlimit(self.limit)
        self.ThreadActive=True
        if self.video_path:
            self.load_xarray_perframe(self.video_path)
            self.get_video()
        else:
            print('No video file selected. Click: Upload Video: to select .avi video file')

    def load_xarray_perframe(self, video_path):
        self.data_array = xr.open_dataset(video_path)
        self.ThreadActive=True


    def get_video(self):
        self.stop_thread = False
        time_frame = 1 / self.frame_rate
        if self.data_array is not None:
            while self.ThreadActive:
                for i in range(self.frame_index, len(self.data_array["frame"])):
                    if not self.stop_thread:
                        img = self.data_array["frame"][i].values  # gets frame i
                        if img is not None:  
                            img = np.uint8(img)  # converts frame to image from pixels
                            if img.size > 0:
                                height, width = img.shape if len(img.shape) >= 2 else (0, 0)  # Handle empty or invalid images
                                if height > 0 and width > 0:
                                    q_img = QImage(img, width, height, width, QImage.Format_Grayscale8)  # passes image to QImage for display
                                    self.updateFrame.emit(q_img)  # emits passed image so that MainWindow can pick it up
                                    time.sleep(time_frame)
                                    self.frame_index = i  # Update frame index
                        else:
                            self.prev_slider_value = self.init_slider_val
                            while self.stop_thread:
                                img = self.data_array["frame"][i].values  # gets frame i
                                if img is not None:
                                    img = np.uint8(img)  # converts frame to image from pixels
                                    if img.size > 0:
                                        height, width = img.shape if len(img.shape) >= 2 else (0, 0)  # Handle empty or invalid images
                                        if height > 0 and width > 0:
                                            q_img = QImage(img, width, height, width, QImage.Format_Grayscale8)  # passes image to QImage for display
                                            self.updateFrame.emit(q_img)  # emits passed image so that MainWindow can pick it up
                                            time.sleep(time_frame)
                                                           

    def update_button_indx(self, button_index):
        self.current_function_index= button_index



    def get_xarray(self):  # Returns the current xarray for saving
        return self.data_array
    
    def set_file_path(self,file_path):
        self.video_path= file_path

    def stop(self): # allows the stop signal from MainWindow to be read by Threading
        self.stop_thread = True # passes stop signal to video loop

    def resume(self): # allows the frame to be paused and resume from last frame
        self.ThreadActive=True
        self.stop_thread = False

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
       
        self.setWindowTitle("xarray_player")
        self.setGeometry(0, 0, 800, 500)

        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        self.upload_button = QPushButton("Upload Video", self)
        self.upload_button.clicked.connect(self.open_file_dialog)
        self.button1 = QPushButton("Start")
        self.button1.clicked.connect(self.start_thread)
        self.button_stop = QPushButton("Pause")
        self.button_stop.clicked.connect(self.stop_thread)

        # Save video
        self.save_video_button = QPushButton("Save Video", self)
        self.save_video_button.clicked.connect(self.save_xarray)        

        layout = QVBoxLayout()
        layout.addWidget(self.label)        
        layout.addWidget(self.upload_button)

        # Stacked Widget for switching between control sets
        self.controlStack = QStackedWidget(self)
        layout.addWidget(self.controlStack)

        layout.addWidget(self.button1)
        layout.addWidget(self.button_stop)
        layout.addWidget(self.save_video_button)
        self.setLayout(layout)

        self.thread = Threading(self)
        self.thread.updateFrame.connect(self.displayFrame)


    @Slot()
    def start_thread(self):
        self.thread.resume()
        self.thread.start()
        self.upload_button.setVisible(False)

    @Slot(QImage)
    def displayFrame(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    @Slot()
    def stop_thread(self):
        self.thread.stop()

    @ Slot()
    def open_file_dialog(self): 
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.nc)")
        if file_path:
            self.thread.set_file_path(file_path)
            self.thread.start()
            self.upload_button.setVisible(False)

    @Slot()
    def save_xarray(self):
        save_path, _ = QFileDialog.getSaveFileName(self, 'Save Video', '', 'NetCDF files (*.nc)')
        if save_path:
            passed_data=self.thread.get_xarray()
            passed_data.to_netcdf(save_path)


            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())