#Main window and UI setup
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QStackedWidget, QCheckBox, QComboBox
from PyQt5.QtCore import Qt
from xarray import DataArray
from PyQt5.QtGui import QImage, QPixmap
import cv2
import video_processing
import numpy as np
from ui_handlers import VideoUIHandlers


class VideoEditorUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Main window settings
        self.setWindowTitle("Video Editor")
        self.setGeometry(100, 100, 800, 600)

        # Central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Video data
        self.video_frames = []
        self.current_frame_index = 0
        self.data_array = None 
        self.ui_handlers = VideoUIHandlers(self)

        # Initialize UI elements
        self.initUI()

    def initUI(self):
        # Video Upload Button
        self.upload_btn = QPushButton("Upload Video", self)
        self.upload_btn.clicked.connect(self.upload_video)
        self.layout.addWidget(self.upload_btn)

        # Video Display Label
        self.video_display = QLabel("Video will be displayed here", self)
        self.layout.addWidget(self.video_display)

        # Stacked Widget for switching between control sets
        self.controlStack = QStackedWidget(self)
        self.layout.addWidget(self.controlStack)

        # Initialize control sets with get_optimal_chk_controls first
        self.init_get_optimal_chk_controls()
        self.init_denoise_controls()
        # Initialize other control sets as needed...
        self.init_motion_estimation_controls()
        self.init_remove_background_controls()
        # And so on for other controls...

        # Current control set index and Next Button setup
        self.current_control_index = 0
        self.next_btn = QPushButton("Next", self)
        self.next_btn.clicked.connect(self.next_control_set)
        self.layout.addWidget(self.next_btn)


    # Next
    def next_control_set(self):
        # Save the video first
        self.ui_handlers.save_video()

        # Move to the next control set
        self.current_control_index += 1
        if self.current_control_index >= self.controlStack.count():
            self.current_control_index = 0
        self.controlStack.setCurrentIndex(self.current_control_index)

    #Get Optimal Controls
    def init_get_optimal_chk_controls(self):
        optimal_chk_widget = QWidget()
        layout = QVBoxLayout(optimal_chk_widget)

        self.get_optimal_chk_btn = QPushButton("Get Optimal Chunk Size", optimal_chk_widget)
        # Connect to the handler in VideoUIHandlers
        self.get_optimal_chk_btn.clicked.connect(self.ui_handlers.handle_get_optimal_chk)

        layout.addWidget(self.get_optimal_chk_btn)
        optimal_chk_widget.setLayout(layout)
        self.controlStack.addWidget(optimal_chk_widget)



    # Denoise Controls
    def init_denoise_controls(self):
        denoise_widget = QWidget()
        denoise_layout = QVBoxLayout(denoise_widget)
        self.denoise_btn = QPushButton("Apply Denoise", denoise_widget)
        self.denoise_slider = QSlider(Qt.Horizontal, denoise_widget)
        # ... [setup the slider] ...

        self.denoise_btn.clicked.connect(self.ui_handlers.apply_denoise) 

        # Connect the signal to the slot here
        self.denoise_btn.clicked.connect(self.ui_handlers.apply_denoise)

        denoise_layout.addWidget(self.denoise_btn)
        denoise_layout.addWidget(self.denoise_slider)

        denoise_widget.setLayout(denoise_layout)
        self.controlStack.addWidget(denoise_widget)



    # Motion Estimation Controls
    def init_motion_estimation_controls(self):
        motion_estimation_widget = QWidget()
        motion_estimation_layout = QVBoxLayout(motion_estimation_widget)
        self.motion_estimation_btn = QPushButton("Estimate Motion", motion_estimation_widget)
        self.motion_estimation_btn.clicked.connect(self.ui_handlers.estimate_motion)
        motion_estimation_layout.addWidget(self.motion_estimation_btn)
        motion_estimation_widget.setLayout(motion_estimation_layout)
        self.controlStack.addWidget(motion_estimation_widget)



    def switch_control_set(self, index):
        self.controlStack.setCurrentIndex(index)



    def init_remove_background_controls(self):
        remove_bg_widget = QWidget()
        remove_bg_layout = QVBoxLayout(remove_bg_widget)

        self.remove_bg_btn = QPushButton("Remove Background", remove_bg_widget)
        # Connect to the method in ui_handlers
        self.remove_bg_btn.clicked.connect(self.ui_handlers.remove_background)

        remove_bg_layout.addWidget(self.remove_bg_btn)

        remove_bg_widget.setLayout(remove_bg_layout)
        self.controlStack.addWidget(remove_bg_widget)



    def init_seed_initialization_controls(self):
        seeds_init_widget = QWidget()
        seeds_init_layout = QVBoxLayout(seeds_init_widget)

        self.init_seeds_btn = QPushButton("Initialize Seeds", seeds_init_widget)
        # Connect to the method in ui_handlers
        self.init_seeds_btn.clicked.connect(self.ui_handlers.initialize_seeds)

        seeds_init_layout.addWidget(self.init_seeds_btn)

        seeds_init_widget.setLayout(seeds_init_layout)
        self.controlStack.addWidget(seeds_init_widget)



    def init_save_video_controls(self):
        save_video_widget = QWidget()
        save_video_layout = QVBoxLayout(save_video_widget)

        self.save_video_btn = QPushButton("Save Video", save_video_widget)
        # Connect to the method in ui_handlers
        self.save_video_btn.clicked.connect(self.ui_handlers.save_video)

        save_video_layout.addWidget(self.save_video_btn)

        save_video_widget.setLayout(save_video_layout)
        self.controlStack.addWidget(save_video_widget)



    def init_update_background_controls(self):
        update_bg_widget = QWidget()
        update_bg_layout = QVBoxLayout(update_bg_widget)

        # Add controls to update_bg_widget
        self.update_bg_btn = QPushButton("Update Background", update_bg_widget)
        self.update_bg_btn.clicked.connect(self.ui_handlers.update_background)

        # Add controls to layout
        update_bg_layout.addWidget(self.update_bg_btn)

        self.controlStack.addWidget(update_bg_widget)

    def init_update_temporal_controls(self):
        update_temporal_widget = QWidget()
        update_temporal_layout = QVBoxLayout(update_temporal_widget)

        # Add controls to update_temporal_widget
        self.update_temporal_btn = QPushButton("Update Temporal", update_temporal_widget)
        self.update_temporal_btn.clicked.connect(self.ui_handlers.update_temporal)

        # Add controls to layout
        update_temporal_layout.addWidget(self.update_temporal_btn)

        self.controlStack.addWidget(update_temporal_widget)



    def init_generate_videos_controls(self):
        generate_videos_widget = QWidget()
        generate_videos_layout = QVBoxLayout(generate_videos_widget)

        # Add controls to generate_videos_widget
        self.generate_videos_btn = QPushButton("Generate Videos", generate_videos_widget)
        self.generate_videos_btn.clicked.connect(self.ui_handlers.generate_videos)

        # Add controls to layout
        generate_videos_layout.addWidget(self.generate_videos_btn)

        self.controlStack.addWidget(generate_videos_widget)



    def init_save_minian_controls(self):
        save_minian_widget = QWidget()
        save_minian_layout = QVBoxLayout(save_minian_widget)

        # Add controls to save_minian_widget
        self.save_minian_btn = QPushButton("Save Minian", save_minian_widget)
        self.save_minian_btn.clicked.connect(self.ui_handlers.save_minian)

        # Add controls to layout
        save_minian_layout.addWidget(self.save_minian_btn)

        self.controlStack.addWidget(save_minian_widget)



    def upload_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi)")
        if file_name:
            self.load_avi_perframe(file_name)
            self.frame_array_2_xarray()
            print("Video converted to xarray. Ready for further processing.")
            self.display_first_frame_or_setup()  # Ensure this is called correctly



    def load_avi_perframe(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Cannot open video.")
                return

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
                    print(f"Error: Cannot read frame {i}.")
                    break
            cap.release()
            self.frame_array = frame_array
            self.stats = [frame_number, height, width]
        except Exception as e:
            print(f"Error during video loading or conversion: {e}")


    def frame_array_2_xarray(self):
        try:
            self.data_array = DataArray(
                self.frame_array,
                dims=["frame", "height", "width"],
                coords={
                    "frame": np.arange(self.stats[0]),
                    "height": np.arange(self.stats[1]),
                    "width": np.arange(self.stats[2]),
                },
            )
            print("Converted to xarray successfully.")
            print(self.data_array) 
        except Exception as e:
            print(f"Error during xarray conversion: {e}")



    def display_first_frame_or_setup(self):
        if self.data_array is not None and len(self.data_array) > 0:
            first_frame = self.data_array[0].values
            # Convert the frame to QImage and display it
            qImg = self.convert_nparray_to_qimage(first_frame)
            self.video_display.setPixmap(QPixmap.fromImage(qImg))
        else:
            print("No data in xarray.")


    def display_frame(self, index):
        if index < len(self.video_frames):
            frame = self.video_frames[index]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.video_display.setPixmap(QPixmap.fromImage(qImg))



    def init_denoise_controls(self):
        denoise_widget = QWidget()
        denoise_layout = QVBoxLayout(denoise_widget)

        # Denoising Method Selection
        denoise_method_label = QLabel("Denoising Method:", self)
        denoise_layout.addWidget(denoise_method_label)
        self.denoise_method_combo = QComboBox(self)
        self.denoise_method_combo.addItems(['gaussian', 'median', 'bilateral'])
        denoise_layout.addWidget(self.denoise_method_combo)

        # Denoising Parameter Slider
        self.denoise_param_label = QLabel("Denoising Parameter: 5", self)
        denoise_layout.addWidget(self.denoise_param_label)
        self.denoise_param_slider = QSlider(Qt.Horizontal, self)
        self.denoise_param_slider.setRange(1, 10)
        self.denoise_param_slider.setValue(5)
        self.denoise_param_slider.valueChanged[int].connect(self.on_denoise_param_change)
        denoise_layout.addWidget(self.denoise_param_slider)

        # Apply Denoise Button
        self.denoise_btn = QPushButton("Apply Denoise", self)
        self.denoise_btn.clicked.connect(self.apply_denoise)
        denoise_layout.addWidget(self.denoise_btn)

        denoise_widget.setLayout(denoise_layout)
        self.controlStack.addWidget(denoise_widget)


    def on_denoise_param_change(self, value):
        self.denoise_param_label.setText(f"Denoising Parameter: {value}")


    def apply_denoise(self):
        if self.current_frame_index < len(self.video_frames):
            frame = self.video_frames[self.current_frame_index]
            denoised_frame = vp.denoise(frame, method='gaussian', kernel_size=5)



    def add_motion_estimation_controls(self):
        self.motion_estimation_btn = QPushButton("Estimate Motion", self)
        self.motion_estimation_btn.clicked.connect(self.ui_handlers.estimate_motion)
        self.layout.addWidget(self.motion_estimation_btn)



    def add_background_removal_controls(self):
        self.remove_background_btn = QPushButton("Remove Background", self)
        self.remove_background_btn.clicked.connect(self.ui_handlers.remove_background)
        self.layout.addWidget(self.remove_background_btn)



    def add_remove_background_controls(self):
        self.remove_bg_btn = QPushButton("Remove Background", self)
        self.remove_bg_btn.clicked.connect(self.ui_handlers.remove_background)
        self.layout.addWidget(self.remove_bg_btn)



    def add_seed_initialization_controls(self):
        self.init_seeds_btn = QPushButton("Initialize Seeds", self)
        self.init_seeds_btn.clicked.connect(self.ui_handlers.initialize_seeds)
        self.layout.addWidget(self.init_seeds_btn)



    def add_save_video_controls(self):
        self.save_video_btn = QPushButton("Save Video", self)
        self.save_video_btn.clicked.connect(self.ui_handlers.save_video)
        self.layout.addWidget(self.save_video_btn)



    def add_update_background_controls(self):
        self.update_bg_btn = QPushButton("Update Background", self)
        self.update_bg_btn.clicked.connect(self.ui_handlers.update_background)
        self.layout.addWidget(self.update_bg_btn)


    def add_update_temporal_controls(self):
        self.update_temporal_btn = QPushButton("Update Temporal", self)
        self.update_temporal_btn.clicked.connect(self.ui_handlers.update_temporal)
        self.layout.addWidget(self.update_temporal_btn)



    def add_generate_videos_controls(self):
        self.generate_videos_btn = QPushButton("Generate Videos", self)
        self.generate_videos_btn.clicked.connect(self.ui_handlers.generate_videos)
        self.layout.addWidget(self.generate_videos_btn)



    def add_save_minian_controls(self):
        self.save_minian_btn = QPushButton("Save Minian", self)
        self.save_minian_btn.clicked.connect(self.ui_handlers.save_minian)
        self.layout.addWidget(self.save_minian_btn)

    def convert_nparray_to_qimage(self, nparray):
        height, width = nparray.shape
        bytesPerLine = width
        return QImage(nparray.data, width, height, bytesPerLine, QImage.Format_Grayscale8)



    
