import sys
import time

import cv2
from PySide6.QtCore import QThread, Signal, Slot, Qt, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QLabel, QFileDialog, QSlider, QStackedWidget, QWidget
from xarray.core.dataarray import DataArray
import numpy as np
import video_processing as vp

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
        self.functions = [self.get_chunk, self.denoise]

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
        self.get_chunk()

    def get_video(self): # This function is responsible for calling video loop and temporarily applying adjustments
        self.slider_value=0
        self.stop_thread = False
        time_frame=1 / self.frame_rate
        if self.data_array is not None:
            while self.ThreadActive:
                for i in range( self.frame_index,len(self.data_array)):  # Start from frame_index
                    if not self.stop_thread:
                        img = self.data_array[i].values # gets frame i
                        if self.slider_value!=0: #checks to see if slider has been adjusted
                                self.prev_slider_value=self.slider_value
                                img = self.data_array[i].values
                                self.frame=img
                                img = self.current_function()
                        img = np.uint8(img) # converts frame to image from pixels
                        height, width = img.shape
                        q_img = QImage(img, width, height, width, QImage.Format_Grayscale8) # passes image to QImage for display
                        self.updateFrame.emit(q_img) # emits passed image so that MainWindow can pick it up
                        time.sleep(time_frame)
                        self.frame_index = i  # Update frame index
                    else: # Ensures adjustments will continue to be seen when paused
                        self.prev_slider_value= 0
                        while self.stop_thread:
                            if self.slider_value!=self.prev_slider_value: #checks to see if brightness has been adjusted
                                self.prev_slider_value=self.slider_value
                                img = self.data_array[i].values
                                self.frame=img
                                img = self.current_function()
                                img = np.uint8(img) # converts frame to image from pixels
                                height, width = img.shape
                                q_img = QImage(img, width, height, width, QImage.Format_Grayscale8) # passes image to QImage for display
                                self.updateFrame.emit(q_img)
                                time.sleep(time_frame)                             

    def update_button_indx(self, button_index):
        self.current_function_index= button_index
        self.current_function = self.functions[self.current_function_index]

    def temp_mod_frame(self, value): # takes value from on_brightness_change and adjusts brightness factor
        # call function based on passed value
        self.slider_value = value


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
        self.apply_changes()
        self.ThreadActive=True # Allows play button to function again

    def apply_changes(self):
        if self.current_function_index!=0:
            for indx in range(len(self.data_array)):
                # change right side based on index
                self.data_array[indx].values=self.current_function(self.data_array[indx].values)
### added functions for calling miniAM video_processing

    def  get_chunk(self):
        self.chunk_comp, self.chunk_store= vp.get_optimal_chk(self.data_array)
        # Note: currently have no idea why we are calling this function, but may be important later

    def denoise(self,frame):
        self.kernal_size=self.slider_value
        vp.denoise(frame, method='gaussian', kernel_size=self.kernel_size)

### will add further functions from ui handlers here and then add them to the current_function array

    def current_function(self, frame):
        return self.current_function(frame)

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.Button_name=['Get optimal chunk', 'Denoise', 'Other'] # Names of sliders
        self.Min_slider=[0, 1,0] # Minimum value for slider
        self.Max_slider=[1,10,1]
        self.init_slider=[0,5,0] # Initial values for sliders
        
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

        # Current control set index and Next Button setup
        self.current_control_index = 0
        self.next_btn = QPushButton("Next", self)
        self.next_btn.clicked.connect(self.next_control_set)

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

### End of miniAM insert
    
        self.thread = Threading(self)
        self.thread.updateFrame.connect(self.displayFrame)



    @ Slot()
    def update_button_index(self):
        button_indx=self.current_control_index
        self.thread.update_button_indx(button_indx) 

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


### Initiating controls for MiniAM
    # Next
    def next_control_set(self):
    # Save the video first
        self.save_changes()
    
    # Move to the next control set
        self.current_control_index += 1
        if self.current_control_index >= len(self.Button_name):
            self.current_control_index = 0

        self.init_current_controls()

    # Create a new button and slider for the next control set
        current_widget = QWidget()
        current_layout = QVBoxLayout(current_widget)

        self.current_button = QPushButton(self.Button_name[self.current_control_index], current_widget)
        self.current_button.clicked.connect(self.current_function) 
        current_layout.addWidget(self.current_button)

        self.current_slider = QSlider(Qt.Horizontal, current_widget)
        self.current_slider.setMinimum(self.Min_slider[self.current_control_index])
        self.current_slider.setMaximum(self.Max_slider[self.current_control_index])
        self.current_slider.setValue(self.init_slider[self.current_control_index])
        current_layout.addWidget(self.current_slider)

        current_widget.setLayout(current_layout)
        self.controlStack.addWidget(current_widget)

        self.controlStack.setCurrentIndex(self.current_control_index)
    
    # Denoise Controls
    def init_current_controls(self):
        current_widget= QWidget()
        current_layout=QVBoxLayout(current_widget)
        self.current_button = QPushButton(self.Button_name[self.current_control_index], current_widget)
        self.current_button.clicked.connect(self.current_function) 
        current_layout.addWidget(self.current_button)
        self.current_slider = QSlider(Qt.Horizontal, current_widget)
        # ... [setup the slider] ...
        current_layout.addWidget(self.current_slider)
        current_widget.setLayout(current_layout)
        self.current_slider.setMinimum(self.Min_slider[self.current_control_index])
        self.current_slider.setMaximum(self.Max_slider[self.current_control_index])
        self.current_slider.setValue(self.init_slider[self.current_control_index])
        self.current_button.clicked.connect(self.save_changes())

    def switch_control_set(self, index):
        self.controlStack.setCurrentIndex(index)

    def current_function(self):
        print('')

    @Slot()
    def on_slider_change(self):
        current_value = self.current_slider.value()
        self.thread.temp_mod_frame(current_value)     # send brightness value to parent
        self.current_label.setText(self.Button_name[self.current_control_index]+": {current_value}")
        self.current_value = current_value

    def add_controls(self):
        self.current_button = QPushButton(self.Button_names[self.current_control_index], self)
        self.current_button.clicked.connect(self) # need to create function
        self.layout.addWidget(self.current_button)
            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())