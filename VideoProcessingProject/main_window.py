#Main window and UI setup
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QStackedWidget, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import video_processing as vp
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

        # Initialize all control sets
        self.init_denoise_controls()
        self.init_motion_estimation_controls()
        self.init_remove_background_controls()
        self.init_seed_initialization_controls()
        self.init_update_background_controls()
        self.init_update_temporal_controls()
        self.init_generate_videos_controls()
        self.init_save_minian_controls()

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
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi *.mp4)")
        if file_name:
            self.load_video(file_name)



    def load_video(self, file_path):
        cap = cv2.VideoCapture(file_path)
        self.video_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.video_frames.append(frame)
        cap.release()
        self.display_frame(0)  # Display the first frame



    def display_frame(self, index):
        if index < len(self.video_frames):
            frame = self.video_frames[index]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.video_display.setPixmap(QPixmap.fromImage(qImg))



    def add_denoise_controls(self):
        self.denoise_btn = QPushButton("Apply Denoise", self)
        self.denoise_slider = QSlider(Qt.Horizontal, self)
        self.denoise_slider.setMinimum(1)
        self.denoise_slider.setMaximum(10)
        self.denoise_slider.setValue(5)
        self.denoise_btn.clicked.connect(self.apply_denoise)
        self.layout.addWidget(self.denoise_btn)
        self.layout.addWidget(self.denoise_slider)



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


    
