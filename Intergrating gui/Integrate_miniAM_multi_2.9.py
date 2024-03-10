import sys
import time

import cv2
from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, QThread, pyqtSignal, pyqtSlot, QRect

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

from xarray import DataArray

# Handles communication to and from subprocesses such as Play_video and Save_changes_2_xarray
# and communication to and from user inputs from MainWindow
class Processing_HUB(QObject):
    updateFrame = pyqtSignal(QImage)
    updateProgressBar = pyqtSignal(int)

    def __init__(self):
        super(Processing_HUB, self).__init__()
        self.video_path = None
        self.data_array = None
        self.play_video_instance = None
        self.current_function_index = 0
        self.deglow = False
        self.data_array=None
        self.current_function_index=0
        self.current_function='None'
        self.decimal_sliders=np.array([13,14,15])
        self.decimal_hundreth=7
        self.decimal_sliders_2=6
        self.init_slider_val=0
        self.slider_value = 0
        self.slider_value_2=None
        self.convert_2_contours=False
        self.init_slider_val_2=None
        self.Truth_array=[]
        self.function_indices_alter_image=np.array([2,3]) # indices of functions that may alter individual frames 
        self.parameters=None
        self.visualize_seeds=False

    def set_file_path(self, file_path):
        self.video_path = file_path

    def get_initial_video(self):
        if self.video_path is not None and len(self.video_path) > 0:
            self.data_array_orig = self.load_video_data()
            self.data_array=self.data_array_orig
            if self.data_array is not None and len(self.data_array) > 0:
                self.play_video_instance = Play_video(self.data_array)
                self.play_video_instance.updateFrame.connect(self.handle_update_frame)
                self.play_video_instance.start()
            print('Video started')
        else:
            print('Failed to get video path')

    def handle_update_frame(self, qimage):
        if self.current_function_index==2 or self.current_function_index==3:
            self.frame_altering_function(frame=qimage) 
        if self.convert_2_contours==True:
            self.convert_contours(frame=qimage)
        if self.visualize_seeds==True:
            self.frame_taking_function(frame=qimage)
            self.show_seeds(qimage)
        self.updateFrame.emit(qimage)

    def handle_xarray_saved(self, data_array):
        self.data_array = data_array
        if self.data_array is not None and len(self.data_array) > 0:
            print('Xarray saved to Processing HUB')

    def stop_play_thread(self):
        if self.play_video_instance:
            self.play_video_instance.stop_thread = True
        print('Video playing is disabled')

    def load_video_data(self):
        initial_file_opening = Initial_file_opening(self.video_path)
        initial_file_opening.xarray_saved.connect(self.handle_xarray_saved)
        initial_file_opening.start()
        initial_file_opening.wait()
        return initial_file_opening.data_array

    def save_changes_2_xarray(self):
        if self.play_video_instance is not None:
            self.play_video_instance.stop_thread = True
        if self.data_array is not None and self.current_function_index<15:
            if len(self.data_array) > 0:
                self.apply_changes_2()
            else:
                print('Data array is empty')
        self.play()

    def play(self):
        if self.play_video_instance is not None and self.play_video_instance.isFinished:
            if self.data_array is not None and len(self.data_array) > 0:
                self.play_video_instance = Play_video(self.data_array)
                self.play_video_instance.updateFrame.connect(self.handle_update_frame)
                self.play_video_instance.start()

    def apply_changes_2(self):  # For functions that require individual frames
        self.generate_parameter_array()
        save_xarray_instance = Save_changes_2_xarray(self.data_array, self.current_function_index, self.parameters)
        save_xarray_instance.updateProgressBar.connect(self.handle_ProgressBar)
        save_xarray_instance.start()
        save_xarray_instance.wait()
        self.data_array = save_xarray_instance.data_array
        return self.data_array

    def handle_ProgressBar(self, integer):
        self.updateProgressBar.emit(integer)

    def update_button_indx(self, button_index):
        self.current_function_index = button_index
        print('The button_index from progress_HUB is ' + str(self.current_function_index))

    def remove_glow(self):
        if self.data_array is not None and len(self.data_array) > 0:
            print('deglow function activated')
            if self.deglow:
                self.play_video_instance.requestInterruption
                self.data_array = self.data_array_orig.sel(frame=slice(None)) - self.data_array.min(dim='frame')  # applies deglow 
                self.play_video_instance.data_array = self.data_array
                self.play()     
        else:
            print('DataArray is empty')

    def glow_check(self):
        self.deglow = True
        print('self.deglow=True')
        self.remove_glow()

    def glow_unchecked(self):
        self.deglow = False
        print('self.deglow=False')
        self.data_array = self.data_array_orig
        self.play_video_instance.data_array = self.data_array

    def show_seeds(self,frame):
        print('Function for showing seeds is active')
    
    def resume(self): # allows the frame to be paused and resume from last frame
        if self.play_video_instance:
            self.play_video_instance.stop_thread = False
    
#### Get current function:
    def frame_altering_function(self,frame):
        if self.current_function_index == 2:
                return self.denoise(frame)
     
        elif self.current_function_index == 3:
                return self.remove_background(frame)

    def frame_taking_function(self,frame):
            if self.current_function_index == 5:
                return self.seeds_init_wrapper(frame)  
            elif self.current_function_index == 6:
                return self.pnr_refine_wrapper(frame)
            elif self.current_function_index == 7:
                return self.ks_refine_wrapper(frame)

   # Not sure were to put this yet                     
    def data_array_only_functions(self):
        if self.current_function_index == 16:
            self.generate_videos_wrapper()

### adjusts parameters to appropriate scale
    def parameters_for_current_functions(self):
        self.new_frame=None
        if self.data_array is not None and len(self.data_array) > 0:
                print('data_array obtained by current function')
        for numb in self.decimal_sliders:
            self.Truth_array.append(numb == self.current_function_index)
        if any(self.Truth_array):
            self.slider_value= (self.slider_value/10)
        if self.current_function_index == self.decimal_hundreth:
            self.slider_value = (self.slider_value / 100)
        if self.current_function_index == self.decimal_sliders_2:
            self.slider_value_2 = (self.slider_value_2 / 10)

    def generate_parameter_array(self):
        self.parameters_for_current_functions()
        if self.slider_value !=0 and self.slider_value_2 is not None:
            self.parameters=np.array([self.slider_value, self.slider_value_2])
        elif self.slider_value!=0:
            self.parameters=np.array([self.slider_value])
        
    def denoise(self, frame):
        frame=self.qimg_2_array(frame)
        if self.slider_value==0:
            kernel_size = 5
        else:
            kernel_size = int(self.slider_value)
        if kernel_size % 2 == 0:
                kernel_size += 1
        new_frame=vp.denoise(frame, method='gaussian', kernel_size=kernel_size)
        return self.array_2_qimg(new_frame)
  
    
    def remove_background(self,frame):
        frame=self.qimg_2_array(frame)
        self.kernel_size=self.slider_value
        new_frame=vp.remove_background(frame, method="uniform", kernel_size=self.kernel_size)
        return self.array_2_qimg(new_frame)
   
    def seeds_init_wrapper(self, frame):
        self.seeds = vp.seeds_init(frame, self.slider_value, self.slider_value_2)
        return frame
    
    def pnr_refine_wrapper(self,frame):
        if self.play_video_instance.isRunning  and hasattr(self, 'seeds') and self.seeds:
            refined_seeds = vp.pnr_refine(frame, self.seeds, self.slider_value, self.slider_value_2)
            self.seeds = refined_seeds
        else:
            print("No frame or seeds available for PNR refinement.")
        return frame

    def ks_refine_wrapper(self, frame): ### May be difficult to display significance level directly
        if self.play_video_instance.isRunning  and hasattr(self, 'seeds') and self.seeds:
            self.seeds = vp.ks_refine(frame, self.seeds, self.slider_value)
        else:
            print("No frame or seeds available for KS refinement.")
        return frame

    ### Contour function for visualization
    def convert_contours(self, frame):
        if frame is not None and len(self.data_array) > 0:
            # self.whiteFrame = 255 * np.ones((frame.shape[0],frame.shape[1],3), np.uint8)
            converted_array=self.qimg_2_array(frame)
            gray = cv2.cvtColor(converted_array, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            img = cv2.Canny(blurred, 10, 100)
            

            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ret, thresh_frame = cv2.threshold(gray_frame, 180, 200, cv2.THRESH_BINARY)
            # contours, her = cv2.findContours(thresh_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # img = cv2.drawContours(self.whiteFrame, contours, -1,(127,127, 127), 1)
        else:
            q_img=frame
            print('Failed to convert frame to contour')
        return q_img
    
    def qimg_2_array(self, frame):
        img = frame.convertToFormat(QImage.Format_RGB888)  # Convert image to RGB format
        width = img.width()
        height = img.height()
        ptr = img.constBits()
        ptr.setsize(img.byteCount())
        
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))  # Reshape as RGB image
        return arr

    def array_2_qimg(self,frame):
        height, width, channel = frame.shape
        if frame is not None:
            q_img = QImage(frame.data, width, height, frame.strides[0], QImage.Format_RGB888)
        return q_img
    
    def contour_check(self):
        self.convert_2_contours=True

    def contour_unchecked(self):
        self.convert_2_contours=False

    ### self defined functions for passing values
    def update_button_indx(self, button_index):
        self.current_function_index= button_index
        self.allow_funct=False
        print('The button_index from Get_current_function is '+str(self.current_function_index))

    def update_data_array(self, data_array):
        self.data_array=data_array

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


class Initial_file_opening(QThread):
    xarray_saved = pyqtSignal(DataArray)

    def __init__(self, file_path):
        super(Initial_file_opening, self).__init__()
        self.first_play = 0
        self.limit = 10 ** 6
        self.video_path = file_path

    def run(self):
        sys.setrecursionlimit(self.limit)
        if self.video_path:
            if self.first_play == 0:
                self.load_avi_perframe(self.video_path)
                self.frame_array_2_xarray()
                print('xarray_generated_from_file')
                self.first_play += 1
        else:
            print('No video file selected. Click: Upload Video: to select .avi video file')

    def load_avi_perframe(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        channel = 3  # 1 for black and white, 3 for red-blue-green
        frame_array = np.empty((frame_number, height, width, 3), dtype=np.uint8)
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
        self.get_chunk()
        self.xarray_saved.emit(self.data_array)
    
    def get_chunk(self):
        if self.data_array is not None and len(self.data_array) > 0:
            self.chunk_comp, self.chunk_store = vp.get_optimal_chk(self.data_array)

# Plays video of Passed self.data_array and updates frame based on current function
class Play_video(QThread):
    updateFrame = pyqtSignal(QImage)

    def __init__(self, data_array):
        super(Play_video, self).__init__()
        self.data_array = data_array
        self.frame_rate = 30
        self.stop_thread = False
        self.convert_2_contours = False
        self.play_index=0 # For get_video progress

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
                                    q_img = QImage(img.data, width, height, img.strides[0], QImage.Format_RGB888)
                                    self.updateFrame.emit(q_img)
                                    time.sleep(time_frame)
                                else:
                                    print('Failed to get frame')   
                                    

# Applies Get_current_function to self.data_array and saves to Processing_HUB using
class Save_changes_2_xarray(QThread):
        updateProgressBar= pyqtSignal(int)
        def __init__(self,data_array, function_index=0, parameters=None):
            super(Save_changes_2_xarray, self).__init__()
            self.data_array=data_array
            self.function_index=function_index
            self.parameters=parameters
            self.slider_value=0
            self.slider_value_2=None
            # self.current_function_array={0: self.return_self(frame), 1: self.return_self(frame),2:self.denoise(frame),3: self.denoise(frame),
                                  # 4: self.remove_background(frame), 5: self.remove_background (frame), 
                                  # 6:self.seeds_init_wrapper(frame), 7:self.ks_refine_wrapper(frame), 
                                  # 8: self.seeds_merge_wrapper(frame), 9: self.initA_wrapper(frame),
                                  # 10: self.initC_wrapper(frame), 11: self.unit_merge_wrapper(frame),
                                  # 12: self.get_noise_fft_wrapper(frame),13: self.update_spatial_wrapper(frame),
                                  # 14: self.update_background_wrapper(frame), 15: self.update_temporal_wrapper(frame)}

        def run(self):
            if self.parameters is not None:
                self.parameters_unpacking()
            if self.data_array is not None and len(self.data_array) > 0:
               self.apply_changes()
            else:
                print('Failed to pass data_array to Save_changes_2_xarray')

        def apply_changes(self):
            array_len = len(self.data_array)
            for indx in range(array_len):
                self.frame_index = indx
                self.updateProgressBar.emit(int(indx / array_len * 100))
                frame = self.data_array[indx].values
                processed_frame = self.current_function(frame)  # Obtain the processed frame
                if processed_frame is not None:
                    self.data_array[indx].values = processed_frame
                else:
                    print('Failed to convert frame ' + str(indx))
            self.frame_index = 0
            print('Changes saved for ' + str(self.function_index))
            return self.data_array
        
        def parameters_unpacking(self):
            self.slider_value=self.parameters[0]
            print('Passed slider value is'+ str(self.slider_value))
            if len(self.parameters)>1:
                self.slider_value_2=self.parameters[1]
            print('2nd passed slider value is'+ str(self.slider_value_2))
        
        def current_function(self, frame):
            # self.current_function_array[int(self.function_index)](frame)
            if self.function_index==0:
                return frame 
            if self.function_index==1:
                return frame
            elif self.function_index == 2:
                return self.denoise(frame)
            elif self.function_index == 3:
                return self.remove_background(frame)
            elif self.function_index == 4:
                return self.apply_transform_2(frame)
            elif self.function_index == 5:
                self.seeds_init_wrapper(frame)  # Assuming this updates internal state and doesn't modify the frame directly
                return frame
            elif self.function_index == 6:
                return self.pnr_refine_wrapper(frame)
            elif self.function_index == 7:
                return self.ks_refine_wrapper(frame)
            elif self.function_index == 8:
                self.seeds_merge_wrapper(frame)  # Again, assuming updates internal state
                return frame
            elif self.function_index == 9:
                return self.initA_wrapper(frame)
            elif self.function_index == 10:
                return self.initC_wrapper(frame)
            elif self.function_index == 11:
                self.unit_merge_wrapper()  # Assuming updates internal state
                return frame
            elif self.function_index == 12:
                return self.get_noise_fft_wrapper(frame)
            elif self.function_index == 13:
                return self.update_spatial_wrapper()
            elif self.function_index == 14:
                return self.update_background_wrapper()
            elif self.function_index == 15:
                return self.update_temporal_wrapper()
            elif self.function_index == 16:
                # self.generate_videos_wrapper()  # This might need special handling
                return frame
            else:
                return frame

        def return_self(self,frame):
            return frame
        
        def denoise(self, frame):
            if self.slider_value==0:
                self.slider_value=5
            kernel_size = int(self.slider_value)
            print('new value selected')
            if kernel_size % 2 == 0:
                    kernel_size += 1
            print('Denoise function called')
            return vp.denoise(frame, method='gaussian', kernel_size=kernel_size)
  
    
        def remove_background(self, frame):
            self.kernel_size=self.slider_value
            return vp.remove_background(frame, method="uniform", kernel_size=self.kernel_size)

        def estimate_motion(self,frame):
            if self.frame_index < len(self.data_array):
                self.previous_frame = self.data_array[self.frame_index-1].values
                self.motion_vector = vp.estimate_motion(self.frame, self.previous_frame)
    
        def apply_transform_2(self, frame):
            self.estimate_motion(self.frame)
            if self.frame_index < len(self.data_array):
                return vp.apply_transform(frame, self.motion_vector, border_mode=cv2.BORDER_REFLECT)

        def seeds_init_wrapper(self, frame):
            self.seeds = vp.seeds_init(frame, self.slider_value, self.slider_value_2)
            return frame

    
        def pnr_refine_wrapper(self, frame):
            if self.frame_index < len(self.data_array) and hasattr(self, 'seeds') and self.seeds:
                refined_seeds = vp.pnr_refine(frame, self.seeds, self.slider_value, self.slider_value_2)
                self.seeds = refined_seeds
            else:
                print("No frame or seeds available for PNR refinement.")
            return frame


        def ks_refine_wrapper(self, frame):
            if self.frame_index < len(self.data_array) and hasattr(self, 'seeds') and self.seeds:
                self.seeds = vp.ks_refine(frame, self.seeds, self.slider_value)
            else:
                print("No frame or seeds available for KS refinement.")
            return frame

    
        def seeds_merge_wrapper(self, frame):
            if hasattr(self, 'seeds') and self.seeds:
                self.seeds = vp.seeds_merge(self.seeds, self.slider_value)
            return frame

        def initA_wrapper(self, frame):
            if self.frame_index < len(self.data_array) and hasattr(self, 'seeds') and self.seeds:
                self.footprints = [vp.initA(frame, seed, self.slider_value) for seed in self.seeds]
            return frame

        def initC_wrapper(self, frame):
            if self.footprints:
                self.C = vp.initC(self.data_array, self.footprints)
            return frame

        def unit_merge_wrapper(self, frame): # May have to run for both initC and initA
            if hasattr(self, 'footprints') and self.footprints:
                self.footprints = vp.unit_merge(self.footprints, self.slider_value)
            return frame

        def get_noise_fft_wrapper(self, frame):
            if self.frame_index < len(self.data_array):
                self.noise_fft = vp.get_noise_fft(frame)
                return frame
            
### Dont need to pass the following functions frames therefore may deal with them seperately without calling loop
        def update_spatial_wrapper(self):
            if hasattr(self, 'footprints') and self.footprints:
                self.footprints = [vp.update_spatial(footprint, self.slider_value) for footprint in self.footprints]

        def update_background_wrapper(self): # the function update background doesn't seem to exist
            if hasattr(self, 'background_components') and self.background_components:
                self.background_components = [vp.update_background(component, self.slider_value) for component in self.background_components]

        def update_temporal_wrapper(self):
            if hasattr(self, 'temporal_components') and self.temporal_components:
                self.temporal_components = vp.update_temporal(self.data_array, self.temporal_components, self.slider_value)


# MainWindow handles all displays/interactions
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
        self.glow_check_counter=0
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
        self.decimal_sliders=np.array([13,14,15])
        self.decimal_hundreth=7
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
        self.glow_checkBox = QCheckBox('Deglow', self)
        self.glow_checkBox.move(680, 460)
        self.glow_checkBox.setVisible(False)
        self.glow_checkBox.stateChanged.connect(self.deglow_checkbox)

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
        self.thread.updateFrame.connect(lambda image: self.displayFrame(image))
        self.thread.updateProgressBar.connect(lambda int: self.updateProgress(int))

#### Need break function for index over 17 ####
## Initiating controls for MiniAM
    # Next
    def next_control_set(self):
        self.next_btn.setVisible(False)
        self.progress.setValue(0)
        self.save_changes()
        self.current_control += 1 
        self.update_button_index()
        if self.current_control==1:
            self.glow_checkBox.setVisible(True)
        if self.current_control==2:
            self.glow_checkBox.setVisible(False)
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
            Truth_array=[]
            for numb in self.decimal_sliders:
                Truth_array.append(numb==cur_index)
            if any(Truth_array):
                initial_slider=(self.init_slider[cur_index]/10)
                self.current_label = QLabel(self.slider_name[cur_index] + ': ' + str(initial_slider), self.current_widget[cur_index])
            if cur_index==self.decimal_hundreth:
                initial_slider=(self.init_slider[cur_index]/100)
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
            default_index = 17
            self.thread.update_button_indx(default_index)

    @pyqtSlot()
    def start_thread(self):
        self.thread.resume()
        self.thread.play()
        self.upload_button.setVisible(False)

    @ pyqtSlot(QImage)
    def displayFrame(self, Image):
        self.label.setPixmap(QPixmap.fromImage(Image))

    @ pyqtSlot()
    def stop_thread(self):
        self.thread.stop_play_thread()

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
        self.thread.allow_xarray_functions()
        self.thread.save_changes_2_xarray()
        self.update_button_index()
        self.progress.setVisible(False)
        self.next_btn.setVisible(True) 
        

    @pyqtSlot()
    def on_slider_change(self):
            current_value = self.current_slider.value()
            Truth_array=[]
            for numb in self.decimal_sliders:
                Truth_array.append(numb==self.current_control)
            if any(Truth_array):
                current_value=(current_value/10)
            if self.current_control==self.decimal_hundreth:
                current_value=(current_value/100)          
            self.thread.temp_mod_frame(current_value)
            self.current_label.setText("{0}: {1}".format(self.slider_name[self.current_control], str(current_value))) 

    @pyqtSlot()
    def on_slider_change_2(self):
            current_value_2=self.current_slider_2.value()
            if self.current_control==self.decimal_sliders_2:
                current_value_2=(current_value_2/10)
            self.thread.temp_mod_frame_2(current_value_2)
            self.current_label_2.setText("{0}: {1}".format(self.slider_name_2[self.intern_indx], str(current_value_2)))  
    
    @pyqtSlot()
    def send_init_slider_val(self):
        self.thread.get_init_val(self.init_slider[self.current_control])
        if self.intern_indx is not None:
            self.thread.get_init_val_2(self.init_slider_2[self.intern_indx])

    @pyqtSlot(int)
    def updateProgress(self, progress_val):
        self.progress.setValue(progress_val)

    @pyqtSlot()
    def countour_checkbox(self):
        if self.check_counter == 0:
            self.thread.contour_check()
            self.check_counter += 1
        else:
            self.thread.contour_unchecked()
            self.check_counter = 0

    @pyqtSlot()
    def deglow_checkbox(self):
        if self.glow_check_counter == 0:
            self.thread.glow_check()
            self.glow_check_counter += 1
        else:
            self.thread.glow_unchecked()
            self.glow_check_counter = 0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())