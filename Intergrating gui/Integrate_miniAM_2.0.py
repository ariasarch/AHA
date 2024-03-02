import sys
import time

import cv2
from PySide6.QtCore import QThread, Signal, Slot, Qt, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QLabel, QFileDialog, QSlider, QStackedWidget, QWidget, QProgressBar
from xarray.core.dataarray import DataArray
import numpy as np
import video_processing as vp

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
        frame_array = np.empty((frame_number, height, width), dtype=np.uint8)
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
        self.deglow()

    def get_video(self):
        self.slider_value = 5
        self.slider_value_2=10
        self.stop_thread = False
        time_frame = 1 / self.frame_rate

        if self.data_array is not None:
            while self.ThreadActive:
                for i in range(self.frame_index, len(self.data_array)):
                    if not self.stop_thread:
                        img = self.data_array[i].values  # gets frame i
                        if self.slider_value != self.init_slider_val:  # checks if slider has been adjusted
                            self.prev_slider_value = self.slider_value
                            img = self.data_array[i].values
                            frame = img
                            img = self.current_function(frame)
                        if img is not None:  # Check if img is not None before converting
                            height, width = img.shape
                            q_img = QImage(img, width, height, width, QImage.Format_Grayscale8)  # passes image to QImage for display
                            self.updateFrame.emit(q_img)  # emits passed image so that MainWindow can pick it up
                            time.sleep(time_frame)
                            self.frame_index = i  # Update frame index
                    else:  # Ensures adjustments will continue to be seen when paused
                        self.prev_slider_value = self.init_slider_val
                        while self.stop_thread:
                            img = self.data_array[i].values  # gets frame i
                            frame = img
                            if self.slider_value != self.prev_slider_value:  # checks if brightness has been adjusted
                                self.prev_slider_value = self.slider_value
                                img = self.current_function(frame)
                            if img is not None:
                                height, width = img.shape
                                q_img = QImage(img, width, height, width, QImage.Format_Grayscale8)  # passes image to QImage for display
                                self.updateFrame.emit(q_img)  # emits passed image so that MainWindow can pick it up
                                time.sleep(time_frame)                                           

    def update_button_indx(self, button_index):
        self.current_function_index= button_index

    def temp_mod_frame(self, value): # takes value from on_brightness_change and adjusts brightness factor
        # call function based on passed value
        self.slider_value = value
        
    def temp_mod_frame_2(self, value_2):
        self.slider_value_2=value_2

    def get_xarray(self):  # Returns the current xarray for saving
        return self.data_array
    
    def get_cur_frame(self):
        return self.frame_index
    
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
        print('changes applied')
        self.ThreadActive=True # Allows play button to function again

    def apply_changes(self):
        if self.current_function_index!=0:
            for indx in range(len(self.data_array)):
                self.frame_index=indx
                new_frame=self.current_function(self.data_array[indx].values)
                if new_frame is not None:  
                    self.data_array[indx].values=new_frame
                else:
                    print('Failed to convert frame '+ str(indx))
        self.frame_index=0
        self.update_button_indx
        self.current_function
        print(self.current_function_index)
        print(self.current_function)
### added functions for calling miniAM video_processing

    def  get_chunk(self):
        self.chunk_comp, self.chunk_store= vp.get_optimal_chk(self.data_array)
        # Note: currently have no idea why we are calling this function, but may be important later

    def deglow(self):
        vp.remove_glow(self.data_array)

    def denoise(self, frame):
        if self.slider_value != self.init_slider_val:
            kernel_size = int(self.slider_value)
            if kernel_size % 2 == 0:
                kernel_size += 1
            return vp.denoise(frame, method='gaussian', kernel_size=kernel_size)
        return frame
    
    def remove_background(self,frame):
        self.kernel_size=self.slider_value
        return vp.remove_background(frame, method="uniform", kernel_size=self.kernel_size)

    def estimate_motion(self, frame):
            if self.frame_index < len(self.data_array):
                self.previous_frame = self.data_array[self.frame_index-1].values
                self.motion_vector = vp.estimate_motion(frame, self.previous_frame)
    
    def apply_transform_2(self, frame):
        self.estimate_motion(frame)
        if self.frame_index < len(self.data_array):
            return vp.apply_transform(frame, self.motion_vector, border_mode=cv2.BORDER_REFLECT)

    def seeds_init_wrapper(self, frame):
        self.seeds = vp.seeds_init(frame, self.slider_value, self.slider_value_2)
        return frame
    
    def pnr_refine_wrapper(self,frame):
        if self.frame_index < len(self.data_array) and hasattr(self, 'seeds') and self.seeds:
            refined_seeds = vp.pnr_refine(frame, self.seeds, self.slider_value, self.slider_value_2)
            self.seeds = refined_seeds
        else:
            print("No frame or seeds available for PNR refinement.")
        return frame

    def ks_refine_wrapper(self,frame):
        if self.frame_index < len(self.data_array) and hasattr(self, 'seeds') and self.seeds:
            # Adjust the significance_level if needed
            self.seeds = vp.ks_refine(frame, self.seeds, self.slider_value)
        else:
            print("No frame or seeds available for KS refinement.")
        return frame
    
    def seeds_merge_wrapper(self,frame):
        if hasattr(self, 'seeds') and self.seeds:
          # Example value, adjust as necessary
            self.seeds = vp.seeds_merge(self.seeds, self.slider_value)
        return frame

    def initA_wrapper(self,frame):
        if self.frame_index < len(self.data_array) and hasattr(self, 'seeds') and self.seeds:
            self.A = [vp.initA(frame, seed, self.slider_value) for seed in self.seeds]
        return frame

    def initC_wrapper(self, frame):
        if hasattr(self, 'A') and self.A:
            self.C = vp.initC(self.data_array, self.A)
        return frame

    def unit_merge_wrapper(self):
        if hasattr(self, 'footprints') and self.footprints:
            self.footprints = vp.unit_merge(self.footprints, self.slider_value)

    def get_noise_fft_wrapper(self, frame):
        if self.frame_index < len(self.data_array):
            self.noise_fft = vp.get_noise_fft(frame)

    def update_spatial_wrapper(self):
        if hasattr(self, 'footprints') and self.footprints:
            self.footprints = [vp.update_spatial(footprint, self.slider_value) for footprint in self.footprints]

    def update_background_wrapper(self):
        if hasattr(self, 'background_components') and self.background_components:
            self.background_components = [vp.update_background(component, self.slider_value) for component in self.background_components]

    def update_temporal_wrapper(self):
        if hasattr(self, 'temporal_components') and self.temporal_components:
            self.temporal_components = vp.update_temporal(self.data_array, self.temporal_components, self.slider_value)

    def generate_videos_wrapper(self):
        if hasattr(self, 'data_array'):
            transformations = [vp.apply_transform]  # Example transformation function, adjust as necessary
            self.generated_videos = vp.generate_videos(self.data_array, transformations)

### will add further functions from ui handlers here and then add them to the current_function array
    def current_function(self, frame):
        if self.current_function_index == 1:
            return self.denoise(frame)
        elif self.current_function_index == 2:
            return self.remove_background(frame)
        elif self.current_function_index == 3:
            return self.apply_transform_2(frame)
        elif self.current_function_index == 4:
            self.seeds_init_wrapper(frame)  # Assuming this updates internal state and doesn't modify the frame directly
            return frame
        elif self.current_function_index == 5:
            return self.pnr_refine_wrapper(frame)
        elif self.current_function_index == 6:
            return self.ks_refine_wrapper(frame)
        elif self.current_function_index == 8:
            self.seeds_merge_wrapper()  # Again, assuming updates internal state
            return frame
        elif self.current_function_index == 8:
            return self.initA_wrapper(frame)
        elif self.current_function_index == 9:
            return self.initC_wrapper(frame)
        elif self.current_function_index == 10:
            self.unit_merge_wrapper()  # Assuming updates internal state
            return frame
        elif self.current_function_index == 11:
            return self.get_noise_fft_wrapper(frame)
        elif self.current_function_index == 12:
            return self.update_spatial_wrapper(frame)
        elif self.current_function_index == 13:
            return self.update_background_wrapper(frame)
        elif self.current_function_index == 14:
            return self.update_temporal_wrapper(frame)
        elif self.current_function_index == 15:
            self.generate_videos_wrapper()  # This might need special handling
            return frame
        else:
            return frame


class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.Button_name = [
            'Get optimal chunk', 'Denoise', 'Remove Background', 'Apply Motion Transform', 
            'Seeds Init', 'PNR Refine', 'KS Refine', 'Seeds Merge', 
            'Init A', 'Init C', 'Unit Merge', 'Get Noise FFT', 
            'Update Spatial', 'Update Background', 'Update Temporal', 'Generate Videos'
        ]
        self.slider_name = [
            'None',  # Get optimal chunk
            'Kernel Size',  # Denoise
            'Kernel Size',  # Remove Background
            'None',  # Apply Transform
            'Threshold',  # Seeds Init
            'Noise Frequency',  # PNR Refine
            'Significance Level',  # KS Refine
            'None',  # Seeds Merge (not applicable, placeholder)
            'Spatial Radius',  # Init A
            'None',  # Init C (not applicable, placeholder)
            'None',  # Unit Merge (not applicable, placeholder)
            'None',  # Get Noise FFT (not applicable, placeholder)
            'Update Factor',  # Update Spatial
            'Update Factor',  # Update Background
            'Update Factor',  # Update Temporal
            'None'  # Generate Videos (not applicable, placeholder)
        ]

        self.Min_slider = [
            0,  # Get optimal chunk
            1,  # Denoise
            1,  # Remove Background
            0,  # Apply Transform
            0,  # Seeds Init (e.g., threshold min)
            1,  # PNR Refine (e.g., min noise frequency)
            0,  # KS Refine (e.g., significance level min)
            0,  # Seeds Merge (not applicable, placeholder)
            1,  # Init A (e.g., spatial radius min)
            0,  # Init C (not applicable, placeholder)
            0,  # Unit Merge (not applicable, placeholder)
            0,  # Get Noise FFT (not applicable, placeholder)
            0,  # Update Spatial (e.g., update factor min)
            0,  # Update Background (e.g., update factor min)
            0,  # Update Temporal (e.g., update factor min)
            0   # Generate Videos (not applicable, placeholder)
        ]

        self.Max_slider = [
            1,  # Get optimal chunk
            10, # Denoise
            10, # Remove Background
            0,  # Apply Transform
            255,# Seeds Init (e.g., threshold max)
            10, # PNR Refine (e.g., max noise frequency)
            1,  # KS Refine (e.g., significance level max)
            0,  # Seeds Merge (not applicable, placeholder)
            10, # Init A (e.g., spatial radius max)
            0,  # Init C (not applicable, placeholder)
            0,  # Unit Merge (not applicable, placeholder)
            0,  # Get Noise FFT (not applicable, placeholder)
            1,  # Update Spatial (e.g., update factor max)
            1,  # Update Background (e.g., update factor max)
            1,  # Update Temporal (e.g., update factor max)
            0   # Generate Videos (not applicable, placeholder)
        ]

        self.init_slider = [
            0,  # Get optimal chunk
            5,  # Denoise
            5,  # Remove Background
            0,  # Apply Transform
            100,# Seeds Init (e.g., initial threshold)
            5,  # PNR Refine (e.g., initial noise frequency)
            0,  # KS Refine (e.g., initial significance level)
            0,  # Seeds Merge (not applicable, placeholder)
            5,  # Init A (e.g., initial spatial radius)
            0,  # Init C (not applicable, placeholder)
            0,  # Unit Merge (not applicable, placeholder)
            0,  # Get Noise FFT (not applicable, placeholder)
            0,  # Update Spatial (e.g., initial update factor)
            0,  # Update Background (e.g., initial update factor)
            0,  # Update Temporal (e.g., initial update factor)
            0   # Generate Videos (not applicable, placeholder)
        ]
        self.current_control = 0 
        self.current_widget = [
            'chnk_widget', 'remove_glow_widget', 'denoise_widget', 'remove_bck_widget', 'Transform_widget', 
            'seeds_init_widget', 'pnr_refine_widget', 'ks_refine_widget', 'seeds_merge_widget', 
            'initA_widget', 'initC_widget', 'unit_merge_widget', 'get_noise_fft_widget', 
            'update_spatial_widget', 'update_background_widget', 'update_temporal_widget', 'generate_videos_widget'
        ]
        self.current_layout = [
            'chnk_layout', 'deglow_widget', 'denoise_layout', 'remove_bck_layout', 'Transform_layout', 
            'seeds_init_layout', 'pnr_refine_layout', 'ks_refine_layout', 'seeds_merge_layout', 
            'initA_layout', 'initC_layout', 'unit_merge_layout', 'get_noise_fft_layout', 
            'update_spatial_layout', 'update_background_layout', 'update_temporal_layout', 'generate_videos_layout'
        ]
        self.slider_name_2=['Minumum seed distance', 'Threshold']
        self.init_slider_2=[10,1.5]
        self.Max_slider_2=[20,3.6]
        self.Min_slider_2=[1,0.1]
        
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
        self.next_btn = QPushButton("Next", self)
        self.next_btn.clicked.connect(self.next_control_set)
        # self.progress = QProgressBar(self)
        # self.progress.setGeometry(200, 80, 250, 20)
        # self.progress.setValue(self.get_frame_index)

        # Save video
        self.save_video_button = QPushButton("Save Video", self)
        self.save_video_button.clicked.connect(self.save_xarray)        

        layout = QVBoxLayout()
        layout.addWidget(self.label)        
        layout.addWidget(self.upload_button)

        # Stacked Widget for switching between control sets
        self.controlStack = QStackedWidget(self)
        layout.addWidget(self.controlStack)

        for i in range(len(self.current_widget)):
            self.current_widget[i]=QWidget()
            self.current_layout[i]=QVBoxLayout(self.current_widget[i])


        layout.addWidget(self.button1)
        layout.addWidget(self.button_stop)
        layout.addWidget(self.save_video_button)
        self.setLayout(layout)

        self.thread = Threading(self)
        self.thread.updateFrame.connect(self.displayFrame)

## Initiating controls for MiniAM
    # Next
    def next_control_set(self):
        self.save_changes()
        self.current_control += 1 
        self.update_button_index()
        if self.current_control >= len(self.Button_name):
            self.current_control=len(self.Button_name) # when we finish we might replace this with a save button or something
        self.controlStack.setCurrentIndex(self.current_control)
        self.init_new_widget(self.current_control)
        
    def init_new_widget(self, cur_index):
        if cur_index>=2:
            self.controlStack.removeWidget(self.current_widget[cur_index-1])
        current_layo=self.current_layout[cur_index] 

        self.current_function_Label = QLabel('{}'.format(self.Button_name[cur_index]), self.current_widget[cur_index])
        current_layo.addWidget(self.current_function_Label)

        self.current_label = QLabel(self.slider_name[cur_index] + ': ' + str(self.init_slider[cur_index]), self.current_widget[cur_index])
        current_layo.addWidget(self.current_label) # Add label for displaying slider value

        self.current_slider = QSlider(Qt.Horizontal, self)
        self.current_slider.valueChanged[int].connect(self.on_slider_change)
        self.current_slider.setMinimum(self.Min_slider[cur_index])
        self.current_slider.setMaximum(self.Max_slider[cur_index])
        self.current_slider.setValue(self.init_slider[cur_index])
        current_layo.addWidget(self.current_slider)

        if cur_index==4 or cur_index==5:
            intern_indx=cur_index-4
            self.current_label_2 = QLabel(self.slider_name_2[intern_indx] + ': ' + str(self.init_slider_2[intern_indx]), self.current_widget[cur_index])
            current_layo.addWidget(self.current_label_2) # Add label for displaying slider value
            self.current_slider_2 = QSlider(Qt.Horizontal, self)
            self.current_slider_2.valueChanged[int].connect(self.on_slider_change_2)
            self.current_slider_2.setMinimum(1)
            self.current_slider_2.setMaximum(21)
            self.current_slider_2.setValue(10)
            current_layo.addWidget(self.current_slider_2)

        self.controlStack.addWidget(self.current_widget[cur_index])

    def switch_control_set(self, index):
        self.controlStack.setCurrentIndex(index)

### End of miniAM insert

    @ Slot()
    def update_button_index(self):
        button_indx=self.current_control
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

    @Slot()
    def on_slider_change(self):
            current_value = self.current_slider.value()
            self.thread.temp_mod_frame(current_value)
            self.current_label.setText("{0}: {1}".format(self.slider_name[self.current_control], str(current_value))) 
    @Slot()
    def on_slider_change_2(self):
            current_value_2=self.current_slider_2.value()
            self.thread.temp_mod_frame_2(current_value_2)
            self.current_label_2.setText("{0}: {1}".format(self.slider_name_2[self.current_control-4], str(current_value_2)))  

    @Slot()
    def get_frame_index(self):
        self.thread.get_cur_frame()          

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())