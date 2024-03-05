import sys
import time

import cv2
from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, QThread, pyqtSignal, pyqtSlot, QRect

# from PySide6.QtGui import QImage, QPixmap
# from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QLabel, QFileDialog, QSlider, QStackedWidget, QWidget, QProgressBar, QCheckBox, QHBoxLayout
from xarray.core.dataarray import DataArray
import numpy as np
import video_processing as vp

sys.setrecursionlimit(10**6)

from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QProgressBar,
    QPushButton, QFileDialog, QWidget, QSlider, QStackedWidget
)
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import cv2
import time
import sys

# Assuming vp is another module in your project, as it is not provided
# Comment or replace the following line with the actual import statement
#import vp

# Importing DataArray from xarray
from xarray import DataArray

class Processing_HUB(QObject):
    updateFrame = pyqtSignal(QImage)
    updateProgressBar= pyqtSignal(int)

    def __init__(self):
        super(Processing_HUB, self).__init__()
        self.video_path = None
        self.data_array = None
        self.play_video_instance = None

    def set_file_path(self, file_path):
        self.video_path = file_path

    def get_initial_video(self):
        if self.video_path is not None and len(self.video_path) > 0:
            # Assuming you have a function 'load_video_data' to load the video data
            self.data_array = self.load_video_data()
            if self.data_array is not None and len(self.data_array) > 0:
                self.play_video_instance = Play_video(self.data_array)
                self.play_video_instance.updateFrame.connect(self.handle_update_frame)
                self.play_video_instance.start()
        else:
            print('Failed to get video path')

    def handle_update_frame(self, qimage):
        self.updateFrame.emit(qimage)

    def stop_play_thread(self):
        if self.play_video_instance:
            self.play_video_instance.stop_thread=True
        print('Video playing is disabled')

    def load_video_data(self):
        self.initial_file_opening = Initial_file_opening(self.video_path)
        self.threadpool = QThreadPool.globalInstance()
        self.threadpool.start(self.initial_file_opening)
        self.threadpool.waitForDone()
        return self.initial_file_opening.data_array

    def save_changes_2_xarray(self):
            self.play_video_instance.stop_thread=True # Stops video feed so that memmory can be used for conversion
            self.play_video_instance.ThreadActive=False # Keeps play button from being activated
            self.data_array = self.apply_changes()
            time.sleep(0.1)
            self.play_video_instance.ThreadActive=True # Allows play button to function again
            if self.play_video_instance.isFinished:
                if self.data_array is not None and len(self.data_array) > 0:
                    self.play_video_instance = Play_video(self.data_array)
                    self.play_video_instance.updateFrame.connect(self.handle_update_frame)
                    self.play_video_instance.start()
        
    def apply_changes(self):
        self.save_xarray_instance=Save_changes_2_xarray(self.data_array)
        self.save_xarray_instance.updateProgressBar.connect(lambda int: self.handle_ProgressBar(int))
        self.save_xarray_instance.start()
        self.save_xarray_instance.wait()
        return self.save_xarray_instance.data_array
        
    def handle_ProgressBar(self, integer):
        self.updateProgressBar.emit(integer)

    # functions below are here to pass to subordinate classes
    def update_button_indx(self, button_index):
        self.current_function_index= button_index
    
class Initial_file_opening(QRunnable):
    
    def __init__(self, file_path):
        super(Initial_file_opening, self).__init__()
        self.first_play = 0
        self.limit = 10 ** 6
        self.video_path = file_path


    def run(self):
        sys.setrecursionlimit(self.limit)
        if self.video_path:
            if self.first_play==0:
                self.load_avi_perframe(self.video_path)
                self.frame_array_2_xarray()
                print('xarray_generated_from_file')
                self.first_play+=1
        else:
            print('No video file selected. Click: Upload Video: to select .avi video file')

    def load_avi_perframe(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        channel=3 # 1 for black and white, 3 for red-blue-green
        frame_array = np.empty((frame_number, height, width, 3), dtype=np.uint8)  # Update the shape to include the color channels
        for i in range(frame_number):
            ret, frame = cap.read()
            if ret:
                frame_conv = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_array[i] = frame_conv
            else:
                break
        self.frame_array = frame_array
        self.stats = [frame_number, height, width, channel]

    def frame_array_2_xarray(self):
        self.data_array = DataArray(
            self.frame_array,
            dims=["frame", "height", "width", "channel"],
            coords={
                "frame": np.arange(self.stats[0]),
                "height": np.arange(self.stats[1]),
                "width": np.arange(self.stats[2]),
                "channel": np.arange(self.stats[3]),
            },
        )
        self.ThreadActive = True
        self.get_chunk()
    
    def get_chunk(self):
        if self.data_array is not None and len(self.data_array) > 0:
            # Assuming you want to pass the first frame, modify as needed
            frame_to_pass = self.data_array
            self.chunk_comp, self.chunk_store = vp.get_optimal_chk(frame_to_pass)

class Play_video(QThread):
    updateFrame = pyqtSignal(QImage)

    def __init__(self, data_array):
        super(Play_video, self).__init__()
        self.data_array = data_array
        self.frame_rate = 30
        self.stop_thread = False
        self.convert_2_contours = False
        self.play_index=0 # For get_video progress
        self.init_slider_val=0
        self.init_slider_val_2= None
        self.convert_2_contours=False
        self.slider_value = 0
        self.slider_value_2=None

    def run(self):
        self.get_video()

    def get_video(self):
        print('Playing')
        time_frame = 1 / self.frame_rate
        if self.data_array is not None:
            for i in range(len(self.data_array)):
                if not self.stop_thread:
                    img = self.data_array[i].values
                    if img is not None:
                        height, width, channel = img.shape
                        frame = img
                        img = self.handle_get_current_function(frame)
                        if img is not None:
                            q_img = QImage(img.data, width, height, img.strides[0], QImage.Format_RGB888)
                            self.updateFrame.emit(q_img)
                            time.sleep(time_frame)
                    else:
                        print('Failed to get frame')
                        continue   
                else:
                        while self.stop_thread:
                                img = self.data_array[i].values
                                if img is not None:
                                    height, width, channel = img.shape
                                    frame = img
                                    img = self.handle_get_current_function(frame)
                                    q_img = QImage(img.data, width, height, img.strides[0], QImage.Format_RGB888)
                                    self.updateFrame.emit(q_img)
                                    time.sleep(time_frame)
                                else:
                                    print('Failed to get frame')
                        else:
                                time.sleep(0.1)  # Sleep briefly to avoid high CPU usage 

    def handle_get_current_function(self, frame):
            self.get_funct_instance = Get_current_function(frame)
            self.threadpool = QThreadPool.globalInstance()
            self.threadpool.start(self.get_funct_instance)
            self.threadpool.waitForDone()
            return self.get_funct_instance.new_frame              

class Save_changes_2_xarray(QThread):
        updateProgressBar= pyqtSignal(int)
        def __init__(self, data_array):
            super(Save_changes_2_xarray, self).__init__()
            self.data_array=data_array

        def run(self):
            if self.data_array is not None and len(self.data_array) > 0:
               self.apply_changes()
            else:
                print('Failed to pass data_array to Save_changes_2_xarray')

        def apply_changes(self):
            array_len=len(self.data_array)
            for indx in range(array_len):
                    self.frame_index=indx
                    self.updateProgressBar.emit(int(indx/array_len*100))
                    new_frame=self.handle_get_current_function(self.data_array[indx].values)
                    if new_frame is not None:  
                        self.data_array[indx].values=new_frame
                    else:
                        print('Failed to convert frame '+ str(indx))
            print('progress_bar' +str(int(indx/array_len*100)))
            self.frame_index=0
            print('Changes saved')
            return self.data_array
        
        def handle_get_current_function(self, frame):
            self.get_funct_instance = Get_current_function(frame)
            self.threadpool = QThreadPool.globalInstance()
            self.threadpool.start(self.get_funct_instance)
            self.threadpool.waitForDone()
            return self.get_funct_instance.new_frame
    
class Get_current_function(QRunnable):
    new_frame = None 
    def __init__(self, frame):
        super(Get_current_function, self).__init__()
        self.frame = frame
        self.data_array = None
        self.current_function_index=0
        self.current_function='None'
        self.decimal_sliders=[7,13,14,15]
        self.decimal_sliders_2=6
        self.init_slider_val=0
        self.init_slider_val_2=None

### selects current function from index
    def run(self):
        if self.current_function_index==any(self.decimal_sliders):
            self.slider_value=(self.slider_value/10)
        if self.current_function_index==self.decimal_sliders_2:
            self.slider_value_2=(self.slider_value_2/10)
        try:
            frame=self.frame
            self.new_frame=self.frame
            if self.init_slider_val != 0:
                if 0 <= self.current_function_index < 17:
                    if self.allow_funct==False:
                        if self.current_function_index == 2:
                            self.new_frame=self.denoise(frame)
                            print('first adjustable function used')       
                        elif self.current_function_index == 3:
                            self.new_frame=self.remove_background(frame)
                        elif self.current_function_index == 5:
                            self.seeds_init_wrapper(frame)  # Assuming this updates internal state and doesn't modify the frame directly
                            self.new_frame=frame
                        elif self.current_function_index == 6:
                            self.new_frame=self.pnr_refine_wrapper(frame)
                        elif self.current_function_index == 7:
                            self.new_frame=self.ks_refine_wrapper(frame)
                        elif self.current_function_index == 9:
                            return self.initA_wrapper(frame)
                        elif self.current_function_index == 13:
                            return self.update_spatial_wrapper()
                        elif self.current_function_index == 14:
                            return self.update_background_wrapper()
                        elif self.current_function_index == 15:
                            return self.update_temporal_wrapper()
                    elif self.allow_funct:
                        if self.current_function_index==1:
                            self.current_function=self.deglow(frame)
                            self.data_a=self.deglow()
                            print('first non-adjustable function used')
                        elif self.current_function_index == 2:
                            self.new_frame=self.denoise(frame)       
                        elif self.current_function_index == 3:
                            return self.remove_background(frame)
                        elif self.current_function_index == 4:
                            return self.apply_transform_2(frame)
                        elif self.current_function_index == 5:
                            self.seeds_init_wrapper(frame)  # Assuming this updates internal state and doesn't modify the frame directly
                            return frame
                        elif self.current_function_index == 6:
                            return self.pnr_refine_wrapper(frame)
                        elif self.current_function_index == 7:
                            return self.ks_refine_wrapper(frame)
                        elif self.current_function_index == 8:
                            self.seeds_merge_wrapper(frame)  # Again, assuming updates internal state
                            return frame
                        elif self.current_function_index == 9:
                            return self.initA_wrapper(frame)
                        elif self.current_function_index == 10:
                            return self.initC_wrapper(frame)
                        elif self.current_function_index == 11:
                            self.unit_merge_wrapper()  # Assuming updates internal state
                            return frame
                        elif self.current_function_index == 12:
                            return self.get_noise_fft_wrapper(frame)
                        elif self.current_function_index == 13:
                            return self.update_spatial_wrapper()
                        elif self.current_function_index == 14:
                            return self.update_background_wrapper()
                        elif self.current_function_index == 15:
                            return self.update_temporal_wrapper()
                        elif self.current_function_index == 16:
                            self.generate_videos_wrapper()  # This might need special handling
                            return frame
        except Exception as e:
                print(f'Current function index not excuted: {e}')
   
    
    # function wrappers:
    def deglow(self, frame):
        new_frame = vp.remove_glow(self.data_array, frame)
        return new_frame
        
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
            if self.save_xarray_instance.frame_index < len(self.data_array):
                self.previous_frame = self.data_array[self.save_xarray_instance.frame_index-1].values
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
            self.footprints = [vp.initA(frame, seed, self.slider_value) for seed in self.seeds]
        return frame

    def initC_wrapper(self, frame):
        if self.footprints:
            self.C = vp.initC(self.data_array, self.footprints)
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

    def update_button_indx(self, button_index):
        self.current_function_index= button_index
        self.allow_funct=False
        print('The current function index is '+str(self.current_function_index))

    def temp_mod_frame(self, value): # takes value from on__change and adjusts value for functions
        # call function based on passed value
        self.slider_value = value
        print('slider value changed in Get_current_function')
        
    def temp_mod_frame_2(self, value_2):
        self.slider_value_2=value_2

    def get_init_val(self,value_init):
        self.init_slider_val=value_init
        print('Initial values obtained')

    def get_init_val_2(self,value_init_2):
        self.init_slider_val_2=value_init_2  

    def allow_xarray_functions(self):
            self.allow_funct=True

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.Button_name = [
            'Get optimal chunk', 'Deglow', 'Denoise', 'Remove Background', 'Apply Motion Transform', 
            'Seeds Init', 'PNR Refine', 'KS Refine', 'Seeds Merge', 
            'Init A', 'Init C', 'Unit Merge', 'Get Noise FFT', 
            'Update Spatial', 'Update Background', 'Update Temporal', 'Generate Videos'
        ]
        self.slider_name = [
            'None',  # Get optimal chunk
            'None',  # Deglow
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
            0,  # Deglow 
            1,  # Denoise
            1,  # Remove Background
            0,  # Apply Transform
            1,  # Seeds Init (e.g., threshold min)
            1,  # PNR Refine (e.g., min noise frequency)    
            1,  # KS Refine (e.g., significance level min)
            0,  # Seeds Merge (not applicable, placeholder)
            1,  # Init A (e.g., spatial radius min)
            0,  # Init C (not applicable, placeholder)
            0,  # Unit Merge (not applicable, placeholder)
            0,  # Get Noise FFT (not applicable, placeholder)
            1,  # Update Spatial (e.g., update factor min) 
            1,  # Update Background (e.g., update factor min)
            1,  # Update Temporal (e.g., update factor min)
            0   # Generate Videos (not applicable, placeholder)
        ]

        self.Max_slider = [
            0,  # Get optimal chunk
            0,  # Deglow widget
            10, # Denoise
            10, # Remove Background
            0,  # Apply Transform
            255,# Seeds Init (e.g., threshold max)
            10, # PNR Refine (e.g., max noise frequency)
            5,  # KS Refine (e.g., significance level max)
            0,  # Seeds Merge (not applicable, placeholder)
            10, # Init A (e.g., spatial radius max)
            0,  # Init C (not applicable, placeholder)
            0,  # Unit Merge (not applicable, placeholder)
            0,  # Get Noise FFT (not applicable, placeholder)
            10,  # Update Spatial (e.g., update factor max)
            10,  # Update Background (e.g., update factor max)
            10,  # Update Temporal (e.g., update factor max)
            0   # Generate Videos (not applicable, placeholder)
        ]
 ### KS Refine, and all 'Update' functions need divide by 10
        self.init_slider = [
            0,  # Get optimal chunk
            0,  # Deglow
            5,  # Denoise
            5,  # Remove Background
            0,  # Apply Transform
            130,# Seeds Init (e.g., initial threshold)
            5,  # PNR Refine (e.g., initial noise frequency)
            1,  # KS Refine (e.g., initial significance level) #### need to divide by ten on reported value and after self.slider in target funct
            0,  # Seeds Merge (not applicable, placeholder)
            5,  # Init A (e.g., initial spatial radius)
            0,  # Init C (not applicable, placeholder)
            0,  # Unit Merge (not applicable, placeholder)
            0,  # Get Noise FFT (not applicable, placeholder)
            1,  # Update Spatial (e.g., initial update factor) #### need to divide by ten on reported value and after self.slider in target funct
            1,  # Update Background (e.g., initial update factor) #### need to divide by ten on reported value and after self.slider in target funct
            1,  # Update Temporal (e.g., initial update factor) #### need to divide by ten on reported value and after self.slider in target funct
            0   # Generate Videos (not applicable, placeholder)
        ]
        self.current_control = 0 
        self.intern_indx=None
        self.current_layo=None
        self.check_counter=0
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
        self.decimal_sliders=[7,13,14,15]
        self.decimal_sliders_2=6
        self.slider_name_2=['Minumum seed distance', 'Threshold']
        self.init_slider_2=[10,15]
        self.Max_slider_2=[20,40]
        self.Min_slider_2=[1,1]
        
        self.setWindowTitle("xarray_player")
        self.setGeometry(0, 0, 800, 500)
        outerLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        Button_layout = QHBoxLayout()

        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        self.upload_button = QPushButton("Upload Video", self)
        self.upload_button.clicked.connect(self.open_file_dialog)
        self.checkBox = QCheckBox('Contours', self)
        self.checkBox.move(680, 480)
        self.checkBox.setVisible(False)
        self.checkBox.stateChanged.connect(self.countour_checkbox)
        self.button_play = QPushButton("Start")
        self.button_play.clicked.connect(self.start_thread)
        self.button_stop = QPushButton("Pause")
        self.button_stop.clicked.connect(self.stop_thread)
        self.button_restart = QPushButton("Replay Video")
        self.button_restart.clicked.connect(self.replay_thread)

        # Current control set index and Next Button setup
        self.next_btn = QPushButton("Next", self)
        self.next_btn.clicked.connect(self.next_control_set)
        self.progress = QProgressBar(self)
        self.progress.setVisible(False)

        # Save video
        self.save_video_button = QPushButton("Save Video", self)
        self.save_video_button.clicked.connect(self.save_xarray)        


        topLayout.addWidget(self.progress)
        topLayout.addWidget(self.label)        
        topLayout.addWidget(self.upload_button)

        # Stacked Widget for switching between control sets
        self.controlStack = QStackedWidget(self)
        topLayout.addWidget(self.controlStack)

        for i in range(len(self.current_widget)):
            self.current_widget[i]=QWidget()
            self.current_layout[i]=QVBoxLayout(self.current_widget[i])
        self.min_widget=QWidget()
        self.thresh_widget=QWidget()
        self.current_widget_2=[self.min_widget,self.thresh_widget]

        Button_layout.addWidget(self.button_restart, 2)
        Button_layout.addWidget(self.button_play,2)
        Button_layout.addWidget(self.button_stop,2)
        Button_layout.addWidget(self.save_video_button,2)

        outerLayout.addLayout(topLayout)
        outerLayout.addLayout(Button_layout)
        
        if self.current_layo != None:
            outerLayout.addLayout(self.current_layo)
        self.setLayout(outerLayout)
    
        # Set the window's main layout

        self.thread= Processing_HUB()
        # self.thread.thread2 = Play_video(self)
        self.thread.thread3 = Save_changes_2_xarray(self)
        self.thread.thread4 =Get_current_function(self)
        self.thread.updateFrame.connect(lambda image: self.displayFrame(image))
        self.thread.updateProgressBar.connect(lambda int: self.updateProgress(int))

## Initiating controls for MiniAM
    # Next
    def next_control_set(self):
        self.next_btn.setVisible(False)
        self.progress.setValue(0)
        self.save_changes()
        self.current_control += 1 
        self.update_button_index()
        if self.current_control >= len(self.Button_name):
            self.current_control=len(self.Button_name) # when we finish we might replace this with a save button or something
        self.controlStack.setCurrentIndex(self.current_control)
        self.init_new_widget(self.current_control)
        self.send_init_slider_val()
        
    def init_new_widget(self, cur_index):
        if cur_index>=2:
            self.controlStack.removeWidget(self.current_widget[cur_index-1])
            self.current_layo.removeWidget(self.current_widget[cur_index-1])
        if 3>cur_index-self.Button_name.index('Seeds Init')>=1:
            self.controlStack.removeWidget(self.current_widget_2[cur_index-self.Button_name.index('Seeds Init')-1])
            self.current_layo.removeWidget(self.current_widget_2[cur_index-self.Button_name.index('Seeds Init')-1])
            print('Widget removed')

        self.current_layo=self.current_layout[cur_index]
        self.current_function_Label = QLabel('{}'.format(self.Button_name[cur_index]), self.current_widget[cur_index])
        self.current_layo.addWidget(self.current_function_Label)

        if self.slider_name[cur_index] != 'None':
            if cur_index==any(self.decimal_sliders):
                initial_slider=(self.init_slider[cur_index]/10)
                self.current_label = QLabel(self.slider_name[cur_index] + ': ' + str(initial_slider), self.current_widget[cur_index])
            else:
                self.current_label = QLabel(self.slider_name[cur_index] + ': ' + str(self.init_slider[cur_index]), self.current_widget[cur_index])
            self.current_layo.addWidget(self.current_label) # Add label for displaying slider value
            self.current_slider = QSlider(Qt.Horizontal, self)
            self.current_slider.valueChanged[int].connect(self.on_slider_change)
            self.current_slider.setMinimum(self.Min_slider[cur_index])
            self.current_slider.setTickInterval(10)
            self.current_slider.setMaximum(self.Max_slider[cur_index])
            self.current_slider.setValue(self.init_slider[cur_index])
            self.current_layo.addWidget(self.current_slider)
            self.current_slider.setEnabled(True)
            self.current_label.setEnabled(True)
        self.controlStack.addWidget(self.current_widget[cur_index])

        if self.Button_name[cur_index] =='Seeds Init' or self.Button_name[cur_index]=='PNR Refine':
            self.intern_indx=cur_index-self.Button_name.index('Seeds Init') # gives an internal index starting at 0 from 'Seeds Init'
            self.current_label_2 = QLabel(self.slider_name_2[self.intern_indx] + ': ' + str(self.init_slider_2[self.intern_indx]), self.current_widget_2[self.intern_indx])
            self.current_layo.addWidget(self.current_label_2) # Add label for displaying slider value
            self.current_slider_2 = QSlider(Qt.Horizontal, self)
            self.current_slider_2.valueChanged[int].connect(self.on_slider_change_2)
            self.current_slider_2.setMinimum(self.Min_slider_2[self.intern_indx])
            self.current_slider_2.setMaximum(self.Max_slider_2[self.intern_indx])
            self.current_slider_2.setTickInterval(10)
            self.current_slider_2.setValue(self.init_slider_2[self.intern_indx])
            self.current_layo.addWidget(self.current_slider_2)
            self.controlStack.addWidget(self.current_widget_2[self.intern_indx])
        else:
            self.intern_indx=None
        self.progress.setVisible(False)
        

    def switch_control_set(self, index):
        self.controlStack.setCurrentIndex(index)

### End of miniAM insert

    @ pyqtSlot()
    def update_button_index(self):
        button_indx=self.current_control
        if 0 <= button_indx < 17:
            self.thread.update_button_indx(button_indx)
        else:
            print('Invalid function index:', button_indx)
            # Handle the invalid index gracefully (e.g., set a default index)
            default_index = 0
            self.thread.update_button_indx(default_index)
   

    @pyqtSlot()
    def start_thread(self):
        self.thread.thread2.resume()
        self.thread.thread2.start()
        self.upload_button.setVisible(False)

    @ pyqtSlot(QImage)
    def displayFrame(self, Image):
        self.label.setPixmap(QPixmap.fromImage(Image))

    @ pyqtSlot()
    def stop_thread(self):
        self.thread.thread2.stop()

    @ pyqtSlot()
    def replay_thread(self):
        self.thread.restart_video() # Still working on this function

    @ pyqtSlot()
    def open_file_dialog(self): 
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi)")
        if file_path:
            self.thread.set_file_path(file_path)
            self.upload_button.setVisible(False)
            self.checkBox.setVisible(True)
            self.send_init_slider_val()  
            self.thread.get_initial_video() 

    @pyqtSlot()
    def save_xarray(self):
        save_path, _ = QFileDialog.getSaveFileName(self, 'Save Video', '', 'NetCDF files (*.nc)')
        if save_path:
            passed_data=self.thread.get_xarray()
            passed_data.to_netcdf(save_path)

    @pyqtSlot()
    def save_changes(self):
        self.progress.setVisible(True)
        self.thread.stop_play_thread()
        self.thread.thread4.allow_xarray_functions()
        self.thread.save_changes_2_xarray()
        self.update_button_index()
        self.progress.setVisible(False)
        self.next_btn.setVisible(True) 
        

    @pyqtSlot()
    def on_slider_change(self):
            current_value = self.current_slider.value()
            if self.current_control==any(self.decimal_sliders):
                current_value=(current_value/10)
            self.thread.thread4.temp_mod_frame(current_value)
            self.current_label.setText("{0}: {1}".format(self.slider_name[self.current_control], str(current_value))) 

    @pyqtSlot()
    def on_slider_change_2(self):
            current_value_2=self.current_slider_2.value()
            if self.current_control==self.decimal_sliders_2:
                current_value_2=(current_value_2/10)
            self.thread.thread4.temp_mod_frame_2(current_value_2)
            self.current_label_2.setText("{0}: {1}".format(self.slider_name_2[self.intern_indx], str(current_value_2)))  
    
    @pyqtSlot()
    def send_init_slider_val(self):
        self.thread.thread4.get_init_val(self.init_slider[self.current_control])
        if self.intern_indx is not None:
            self.thread.thread4.get_init_val_2(self.init_slider_2[self.intern_indx])

    @pyqtSlot(int)
    def updateProgress(self, progress_val):
        self.progress.setValue(progress_val)

    @pyqtSlot()
    def countour_checkbox(self):
        self.thread.thread4.contour_check()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())