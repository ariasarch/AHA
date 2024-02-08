import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, \
    QPushButton, QFileDialog, QSlider, QHBoxLayout, QScrollArea
from PyQt5.QtGui import QPixmap, QImage, QColor, QPageSize
from PyQt5.QtCore import Qt, QTimer
import cv2


class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.current_frame = None  # Store the current frame
        self.brightness_value = 50  # Initial brightness value
        self.contrast_value = 50  # Initial contrast value
        self.saturation_value = 0  # Initial saturation value
        self.playing= False 
        self.speed_factor = 2.0 # Initial normalized speed

        self.setWindowTitle('new gui') # overall window label
        ########
        outerLayout = QVBoxLayout() # Outter most layout

        ### Video label and display
        #Vid_lab_layout=QHBoxLayout() # Top most inner nested layout
        #outerLayout.setStretch(0,-100)
        # Edited video label
        #self.edit_label = QLabel("Edited_video:", self)
        #Vid_lab_layout.addWidget(self.edit_label)

        # Original video label
        #self.original_label = QLabel("Original_video:", self)
        #Vid_lab_layout.addWidget(self.original_label)

        Vid_layout = QHBoxLayout() # Second topmost nested layout

        # Dispaly area for edited video
        self.label = QLabel(self)
        Vid_layout.addWidget(self.label)

        # Dispaly area for original video
        self.original_label = QLabel(self)
        Vid_layout.addWidget(self.original_label)

        ### Helpful buttons for loading and playing video:
        Toplayout = QHBoxLayout() # Third topmost inner nested layout

        # Open video file button
        self.select_button = QPushButton('Select Video File', self)
        self.select_button.clicked.connect(self.open_video_file)
        Toplayout.addWidget(self.select_button,3)
        # Play button
        self.play_button = QPushButton('Play', self)
        self.play_button.clicked.connect(self.play)
        Toplayout.addWidget(self.play_button,2)
        # Stop button
        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop)
        Toplayout.addWidget(self.stop_button,2)
        # Save button
        self.save_button = QPushButton('Save Video', self)
        self.save_button.clicked.connect(self.save_video) # Clicked button calls save_video
        Toplayout.addWidget(self.save_button,2)
        
        # Slider nested layout
        slider_layout=QVBoxLayout()

        ### Speed Label and Slider layout
        self.speed_label = QLabel("Speed:", self)
        slider_layout.addWidget(self.speed_label)
        self.speed_slider = QSlider(Qt.Horizontal, self)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.change_speed)
        slider_layout.addWidget(self.speed_slider)

        # Brightness Label and Slider
        self.brightness_label = QLabel("Brightness", self)
        slider_layout.addWidget(self.brightness_label)
        self.brightness_slider = QSlider(Qt.Horizontal, self)
        self.brightness_slider.valueChanged.connect(self.on_brightness_change)
        slider_layout.addWidget(self.brightness_slider)

        # Contrast Label and Slider
        self.contrast_label = QLabel("Contrast", self)
        slider_layout.addWidget(self.contrast_label)
        self.contrast_slider = QSlider(Qt.Horizontal, self)
        self.contrast_slider.valueChanged.connect(self.on_contrast_change)
        slider_layout.addWidget(self.contrast_slider)

        # Saturation Label and Slider
        self.saturation_label = QLabel("Saturation", self)
        slider_layout.addWidget(self.saturation_label)
        self.saturation_slider = QSlider(Qt.Horizontal, self)
        self.saturation_slider.valueChanged.connect(self.on_saturation_change)
        slider_layout.addWidget(self.saturation_slider)

        # Adds nested layouts
        # outerLayout.addLayout(Vid_lab_layout)
        outerLayout.addLayout(Vid_layout)
        outerLayout.addLayout(Toplayout)
        outerLayout.addLayout(slider_layout)
        # Set the window's main layout
        self.setLayout(outerLayout)

        # Sets play time for both videos
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Set window properties
        self.setWindowTitle("Video Editor")
        self.setGeometry(600, 100, 1000, 800)


        # Set ranges for sliders
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(self.brightness_value)

        self.contrast_slider.setRange(0, 100)
        self.contrast_slider.setValue(self.contrast_value)

        self.saturation_slider.setRange(0, 100)
        self.saturation_slider.setValue(self.saturation_value)


    ### Defines functions to be called by buttons or sliders:

    def on_brightness_change(self, value):
        self.brightness_value = value

    def on_contrast_change(self, value):
        self.contrast_value = value

    def on_saturation_change(self, value):
        self.saturation_value = value

    def adjust_brightness(self, frame, value):
        brightness_factor = (value - 50) / 25.0
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness_factor) # saves new global brightness to frame
        return frame

    def adjust_contrast(self, frame, value):
        contrast_factor = 1 + (value - 50) / 50.0
        frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=0) # saves new global contrast to frame
        return frame

    def adjust_saturation(self, frame, value):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # opens color system
        h, s, v = cv2.split(hsv) # grabs s = saturation

        saturation_factor = (value - 50) / 50.0 # new saturation number
        s = cv2.convertScaleAbs(s, alpha=saturation_factor, beta=0) # repackages saturation

        hsv = cv2.merge([h, s, v]) # merges color system
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) # saves new global color saturation toframe
        return frame
    
    def change_speed(self, value):
        self.speed_factor = value / 100.0 # adjusts speed factor
    
### Button functions
    def open_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Video File', '', 'Video files (*.mp4 *.avi *.mkv)')
        if file_path:
            self.play_video(file_path)
    def play(self):
        if self.cap is None:
            return
        if not self.playing:
            self.timer.start(500//int(self.speed_factor)) # adjusts global video speed based on speed factor
            self.playing = True

    def stop(self):
        self.timer.stop() # tells timer to stop
        self.cap.release() # shut down capture system
        self.playing = False

# Save video button function      
    def save_video(self):
        if self.current_frame is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Video', '', 'Video files (*.avi)')
        if file_path:
            height, width, _ = self.current_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(file_path, fourcc, 30.0, (width, height))

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                out.write(frame) # Writes save

            out.release() # releases writing 
            self.cap.release() # releases capture

### Frame adjustments and playback
    def update_frame(self): # updates frame with global values
        if self.cap is None or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame")
            return

        self.current_frame = frame
        processed_frame = self.process_frame(frame) # calls process frame for current frame
        self.display_frame(processed_frame)

    # applies global values to edited video frames
    def process_frame(self, frame):
        # Apply brightness adjustment
        frame = self.adjust_brightness(frame, self.brightness_value)

        # Apply contrast adjustment
        frame = self.adjust_contrast(frame, self.contrast_value)

        # Apply saturation adjustment
        frame = self.adjust_saturation(frame, self.saturation_value)

        return frame
    
    # Displays both edited and original videos
    def display_frame(self, frame):
        # Convert the frame to RGB format and set it as the label's pixmap for edited video
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)

        # Display original video frame
        if self.cap is not None and self.cap.isOpened():
            current_frame=cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(current_frame.data, self.current_frame.shape[1], self.current_frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.original_label.setPixmap(pixmap)

    # Sets video capture which allows the videos to play
    def play_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if self.cap.isOpened():
            self.timer.start(30) # short delay to avoid issues

if __name__ == '__main__':
    app = QApplication(sys.argv) # Passes system arguments to OS so that it can tell it to open a window
    player = VideoPlayer() # sets variable player to widget
    player.show() # shows window for videoplayer
    sys.exit(app.exec_()) # allows you to exit window