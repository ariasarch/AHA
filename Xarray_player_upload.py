import sys
import time

import cv2
from PySide6.QtCore import QThread, Signal, Slot, Qt, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QLabel, QFileDialog, QSlider
from xarray.core.dataarray import DataArray
import numpy as np

class Threading(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame_rate = 30  # Frames per second
        self.data_array = None
        self.stats = None
        self.stop_thread = False
        self.frame_index = 0  

    def run(self):
        if self.video_path:
            self.load_avi_perframe(self.video_path)
            self.frame_array_2_xarray()
            self.get_video()
        else:
            print('No video file selected. Click: Upload Video: to select .avi video file')

    def load_avi_perframe(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_array = np.empty((frame_number, height, width), dtype=np.float64)
        for i in range(frame_number):
            ret, frame = cap.read()
            if ret:
                frame_conv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_array[i] = frame_conv 
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
        self.ThreadActive=True
        

    def get_video(self):
        self.brightness_factor=0
        self.stop_thread = False
        time_frame=1 / self.frame_rate
        if self.data_array is not None:
            while self.ThreadActive:
                for i in range( self.frame_index,len(self.data_array)):  # Start from frame_index
                    if not self.stop_thread:
                        img = self.data_array[i].values # gets frame i
                        if self.brightness_factor!=0: #checks to see if brightness has been adjusted
                            img = cv2.convertScaleAbs(img, alpha=1, beta=self.brightness_factor)
                        img = np.uint8(img) # converts frame to image from pixels
                        height, width = img.shape
                        q_img = QImage(img, width, height, width, QImage.Format_Grayscale8) # passes image to QImage for display
                        self.updateFrame.emit(q_img) # emits passed image so that MainWindow can pick it up
                        time.sleep(time_frame)
                        self.frame_index = i  # Update frame index
                    else:
                        break

    def temp_mod_frame(self, value): # takes value from on_brightness_change and adjusts brightness factor
        # using brightness as proof of concept
        self.brightness_factor = (value - 50) / 5.0
    

    # def apply_mod_ frame:
            # modifies:
            # self.data_array

    def get_xarray(self):  # Returns the current xarray for saving
        return self.data_array
    
    def set_file_path(self,file_path):
        self.video_path= file_path

    def stop(self): # allows the stop signal from MainWindow to be read by Threading
        self.stop_thread = True # passes stop signal to video loop

    def resume(self): # allows the frame to be paused and resume from last frame
        self.ThreadActive=True
        self.stop_thread = False

    def apply_mod_2_xarray(self): # takes current temp_mod_frame parrameter and applies it to entire array
        self.stop_thread=True # Stops video feed so that memmory can be used for conversion
        self.ThreadActive=False # Keeps play button from being activated
        for indx in range(len(self.data_array)):
            self.data_array[indx].values=cv2.convertScaleAbs(self.data_array[indx].values, alpha=1, beta=self.brightness_factor)
        self.ThreadActive=True # Allows play button to function again


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
        self.save_video_button = QPushButton("Save Changes", self)
        self.save_video_button.clicked.connect(self.save_changes)
        self.save_video_button = QPushButton("Save Video", self)
        self.save_video_button.clicked.connect(self.save_xarray)        

        layout = QVBoxLayout()
        layout.addWidget(self.label)        
        layout.addWidget(self.upload_button)

        self.brightness_label = QLabel("Brightness: 50", self)
        layout.addWidget(self.brightness_label)

        self.brightness_slider = QSlider(Qt.Horizontal, self)
        self.brightness_slider.valueChanged[int].connect(self.on_brightness_change)
        layout.addWidget(self.brightness_slider)
        layout.addWidget(self.button1)
        layout.addWidget(self.button_stop)
        layout.addWidget(self.save_video_button)
        self.setLayout(layout)
    

        self.thread = Threading(self)
        self.thread.updateFrame.connect(self.displayFrame)

                # Brightness slider
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(50)   

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
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi)")
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


    @Slot()
    def save_changes(self):
        self.thread.apply_mod_2_xarray()
    
    @Slot()
    def on_brightness_change(self):
        brightness_value = self.brightness_slider.value()
        self.thread.temp_mod_frame(brightness_value)     # send brightness value to parent
        self.brightness_label.setText(f"Brightness: {brightness_value}")
        self.brightness_value = brightness_value
            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())