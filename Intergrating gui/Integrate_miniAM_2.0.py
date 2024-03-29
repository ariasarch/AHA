import sys
import time

import cv2
from PySide6.QtCore import QThread, Signal, Slot, Qt, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QLabel, QFileDialog, QSlider, QStackedWidget, QWidget, QProgressBar, QCheckBox, QHBoxLayout
from xarray.core.dataarray import DataArray
import numpy as np
import video_processing as vp
import threading as Thread

sys.setrecursionlimit(10**6)

class Threading(QThread):
    updateFrame = Signal(QImage)
    updateProgressBar= Signal(int)
    

    def __init__(self, parent=None):

        super().__init__(parent)
        self.frame_rate = 30  # Frames per second
        self.data_array = None
        self.stats = None
        self.stop_thread = False
        self.frame_index = 0  # For progress bar
        self.play_index=0 # For get_video progress
        self.button_index= 0 
        self.slider_value = 0
        self.slider_value_2 = None
        self.current_function_index = 0
        self.limit=10**6
        self.init_slider_val=0
        self.init_slider_val_2= None
        self.convert_2_contours=False
        self.footprints=None
        self.first_play=0
        

    def run(self):
        sys.setrecursionlimit(self.limit)
        if self.video_path:
            if self.first_play==0:
                self.load_avi_perframe(self.video_path)
                self.frame_array_2_xarray()
                print('xarray_generated_from_file')
                self.first_play+=1
            self.get_video()
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
        self.whiteFrame = 255 * np.ones((height,width,3), np.uint8)

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
        self.ThreadActive=True
        self.get_chunk()

    def get_video(self):
        self.stop_thread = False
        time_frame = 1 / self.frame_rate
        if self.data_array is not None:
            while self.ThreadActive:
                for i in range(self.play_index, len(self.data_array)):
                    if not self.stop_thread:
                        img = self.data_array[i].values
                        if self.init_slider_val != 0:
                            frame = img
                            img = self.current_function(frame)
                        if img is not None:
                            height, width, channel = img.shape
                            if self.convert_2_contours==True:
                                img=self.convert_contours(img)
                            q_img = QImage(img, width, height, QImage.Format_RGB888)
                            self.updateFrame.emit(q_img)
                            time.sleep(time_frame)
                            self.play_index = i
                        else:
                            break
                    else:
                        self.prev_slider_value = self.init_slider_val
                        self.prev_slider_value_2 = self.init_slider_val_2
                        while self.stop_thread:
                            if self.slider_value != self.prev_slider_value or self.slider_value_2 != self.prev_slider_value_2:
                                self.prev_slider_value = self.slider_value
                                self.prev_slider_value_2 = self.slider_value_2
                                img = self.data_array[i].values
                                frame = img
                                img = self.current_function(frame)
                            if img is not None:
                                height, width, channel = img.shape
                                if self.convert_2_contours==True:
                                    img=self.convert_contours(img)
                                q_img = QImage(img, width, height, QImage.Format_RGB888)
                                self.updateFrame.emit(q_img)
                                time.sleep(time_frame)      
    def get_video_2(self):
            self.stop_thread = False
            time_frame = 1 / self.frame_rate
            if self.data_array is not None:
                while self.ThreadActive:
                    for i in range(self.play_index, len(self.data_array)):
                        if not self.stop_thread:
                            img = self.data_array[i].values
                            if self.init_slider_val != 0:
                                frame = img
                                img = self.current_function(frame)
                            if img is not None:
                                height, width, channel = img.shape
                                if self.convert_2_contours==True:
                                    img=self.convert_contours(img)
                                q_img = QImage(img, width, height, QImage.Format_RGB888)
                                self.updateFrame.emit(q_img)
                                time.sleep(time_frame)
                                self.play_index = i
                            else:
                                break
                        else:
                            self.prev_slider_value = self.init_slider_val
                            self.prev_slider_value_2 = self.init_slider_val_2
                            while self.stop_thread:
                                if self.slider_value != self.prev_slider_value or self.slider_value_2 != self.prev_slider_value_2:
                                    self.prev_slider_value = self.slider_value
                                    self.prev_slider_value_2 = self.slider_value_2
                                    img = self.data_array[i].values
                                    frame = img
                                    img = self.current_function(frame)
                                if img is not None:
                                    height, width, channel = img.shape
                                    if self.convert_2_contours==True:
                                        img=self.convert_contours(img)
                                    q_img = QImage(img, width, height, QImage.Format_RGB888)
                                    self.updateFrame.emit(q_img)
                                    time.sleep(time_frame)                                   

    def update_button_indx(self, button_index):
        self.current_function_index= button_index

    def temp_mod_frame(self, value): # takes value from on__change and adjusts value for functions
        # call function based on passed value
        self.slider_value = value
        
    def temp_mod_frame_2(self, value_2):
        self.slider_value_2=value_2

    def get_init_val(self,value_init):
        self.slider_value=value_init
        self.init_slider_val=value_init

    def get_init_val_2(self,value_init_2):
        self.slider_value_2=value_init_2
        self.init_slider_val_2=value_init_2

    def get_xarray(self):  # Returns the current xarray for saving
        return self.data_array
    
    def set_file_path(self,file_path):
        self.video_path= file_path

    def stop(self): # allows the stop signal from MainWindow to be read by Threading
        self.stop_thread = True # passes stop signal to video loop

    def resume(self): # allows the frame to be paused and resume from last frame
        self.ThreadActive=True
        self.stop_thread = False

    def restart_video(self):
        if self.play_index==(len(self.data_array)-1):
            self.play_index=0
            time.sleep(0.3)
            self.stop_thread = False
            self.ThreadActive=True
            replay=Thread(target=self.get_video_2())
            replay.start()
            print('restarting')
        else:
            print('Video cannot be restarted until video has ended.')

    def apply_mod_2_xarray(self): # takes current temp_mod_frame parrameter and applies it to entire array
        self.stop_thread=True # Stops video feed so that memmory can be used for conversion
        self.ThreadActive=False # Keeps play button from being activated
        self.apply_changes()
        print('changes applied')
        self.ThreadActive=True # Allows play button to function again

    def apply_changes(self):
        if self.current_function_index!=0:
            array_len=len(self.data_array)
            for indx in range(array_len):
                self.frame_index=indx
                self.updateProgressBar.emit(int(indx/array_len*100))
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
    
    def contour_check(self):
        self.convert_2_contours=True

    def contour_unchecked(self):
        self.convert_2_contours=False

    def convert_contours(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh_frame = cv2.threshold(gray_frame, 180, 200, cv2.THRESH_BINARY)
        contours, her = cv2.findContours(thresh_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        img = cv2.drawContours(self.whiteFrame, contours, -1,(127,127, 127), 1)
        return img


    def get_chunk(self):
        if self.data_array is not None and len(self.data_array) > 0:
            # Assuming you want to pass the first frame, modify as needed
            frame_to_pass = self.data_array
            self.chunk_comp, self.chunk_store = vp.get_optimal_chk(frame_to_pass)

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

### will add further functions from ui handlers here and then add them to the current_function array
    def current_function(self, frame):
        if self.current_function_index==0:
            return frame 
        if self.current_function_index==1:
            return self.deglow(frame)
        elif self.current_function_index == 2:
            return self.denoise(frame)
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
        else:
            return frame


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
            0,  # Seeds Init (e.g., threshold min)
            0.5,  # PNR Refine (e.g., min noise frequency)
            0,  # KS Refine (e.g., significance level min)
            0,  # Seeds Merge (not applicable, placeholder)
            1,  # Init A (e.g., spatial radius min)
            0,  # Init C (not applicable, placeholder)
            0,  # Unit Merge (not applicable, placeholder)
            0,  # Get Noise FFT (not applicable, placeholder)
            0.1,  # Update Spatial (e.g., update factor min)
            0.1,  # Update Background (e.g., update factor min)
            0.1,  # Update Temporal (e.g., update factor min)
            0   # Generate Videos (not applicable, placeholder)
        ]

        self.Max_slider = [
            1,  # Get optimal chunk
            1,  # Deglow widget
            10, # Denoise
            10, # Remove Background
            0,  # Apply Transform
            255,# Seeds Init (e.g., threshold max)
            10, # PNR Refine (e.g., max noise frequency)
            0.5,  # KS Refine (e.g., significance level max)
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
            0,  # Deglow
            5,  # Denoise
            5,  # Remove Background
            0,  # Apply Transform
            130,# Seeds Init (e.g., initial threshold)
            5,  # PNR Refine (e.g., initial noise frequency)
            0.1,  # KS Refine (e.g., initial significance level)
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

        self.slider_name_2=['Minumum seed distance', 'Threshold']
        self.init_slider_2=[10,1.5]
        self.Max_slider_2=[20,3.6]
        self.Min_slider_2=[1,0.1]
        
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

        self.thread = Threading(self)
        self.thread.updateFrame.connect(self.displayFrame)
        self.thread.updateProgressBar.connect(self.updateProgress)

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
        if self.Button_name[cur_index-1]=='PNR Refine' and self.intern_indx!=None:
            self.controlStack.removeWidget(self.current_widget_2[self.intern_indx-1])
        self.current_layo=self.current_layout[cur_index] 

        self.current_function_Label = QLabel('{}'.format(self.Button_name[cur_index]), self.current_widget[cur_index])
        self.current_layo.addWidget(self.current_function_Label)

        if self.slider_name[cur_index] != 'None':
            self.current_label = QLabel(self.slider_name[cur_index] + ': ' + str(self.init_slider[cur_index]), self.current_widget[cur_index])
            self.current_layo.addWidget(self.current_label) # Add label for displaying slider value

            self.current_slider = QSlider(Qt.Horizontal, self)
            self.current_slider.valueChanged[int].connect(self.on_slider_change)
            self.current_slider.setMinimum(self.Min_slider[cur_index])
            if self.Min_slider[cur_index]<10:
                self.current_slider.setTickInterval(0.1)
            self.current_slider.setMaximum(self.Max_slider[cur_index])
            self.current_slider.setValue(self.init_slider[cur_index])
            self.current_layo.addWidget(self.current_slider)
            self.current_slider.setEnabled(True)
            self.current_label.setEnabled(True)

        if self.Button_name[cur_index] =='Seeds Init' or self.Button_name[cur_index]=='PNR Refine':
            self.intern_indx=cur_index-self.Button_name.index('Seeds Init') # gives an internal index starting at 0 from 'Seeds Init'
            self.current_label_2 = QLabel(self.slider_name_2[self.intern_indx] + ': ' + str(self.init_slider_2[self.intern_indx]), self.current_widget_2[self.intern_indx])
            self.current_layo.addWidget(self.current_label_2) # Add label for displaying slider value
            self.current_slider_2 = QSlider(Qt.Horizontal, self)
            self.current_slider_2.valueChanged[int].connect(self.on_slider_change_2)
            self.current_slider_2.setMinimum(self.Min_slider_2[self.intern_indx])
            self.current_slider_2.setMaximum(self.Max_slider_2[self.intern_indx])
            if self.Max_slider_2[self.intern_indx]<10:
                self.current_slider_2.setTickInterval(10)
            self.current_slider_2.setValue(self.init_slider_2[self.intern_indx])
            self.current_layo.addWidget(self.current_slider_2)
            self.controlStack.addWidget(self.current_widget_2[self.intern_indx])
        else:
            self.intern_indx=None

        self.progress.setVisible(False)
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
    def replay_thread(self):
        self.thread.restart_video()

    @ Slot()
    def open_file_dialog(self): 
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi)")
        if file_path:
            self.thread.set_file_path(file_path)
            self.thread.start()
            self.upload_button.setVisible(False)
            self.checkBox.setVisible(True)

    @Slot()
    def save_xarray(self):
        save_path, _ = QFileDialog.getSaveFileName(self, 'Save Video', '', 'NetCDF files (*.nc)')
        if save_path:
            passed_data=self.thread.get_xarray()
            passed_data.to_netcdf(save_path)

    @Slot()
    def save_changes(self):
        self.progress.setVisible(True)
        self.thread.apply_mod_2_xarray()
        self.update_button_index()
        self.progress.setVisible(False)
        self.next_btn.setVisible(True) 
        

    @Slot()
    def on_slider_change(self):
            current_value = self.current_slider.value()
            self.thread.temp_mod_frame(current_value)
            self.current_label.setText("{0}: {1}".format(self.slider_name[self.current_control], str(current_value))) 
    @Slot()
    def on_slider_change_2(self):
            current_value_2=self.current_slider_2.value()
            self.thread.temp_mod_frame_2(current_value_2)
            self.current_label_2.setText("{0}: {1}".format(self.slider_name_2[self.intern_indx], str(current_value_2)))  
    
    @Slot()
    def send_init_slider_val(self):
        self.thread.get_init_val(self.init_slider[self.current_control])
        if self.intern_indx is not None:
            self.thread.get_init_val_2(self.init_slider_2[self.intern_indx])

    @Slot(int)
    def updateProgress(self, progress_val):
        self.progress.setValue(progress_val)

    @Slot()
    def countour_checkbox(self):
        if self.check_counter == 0:
            self.thread.contour_check()
            self.check_counter += 1
        else:
            self.thread.contour_unchecked()
            self.check_counter = 0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())