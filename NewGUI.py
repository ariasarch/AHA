import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QSlider, QFileDialog, QPushButton, QComboBox

class VideoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.video_path = None
        self.current_frame = None
        self.brightness_value = 50
        self.contrast_value = 50
        self.saturation_value = 50
        self.initUI()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.setAcceptDrops(True)


    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setAcceptDrops(True)
        self.layout = QVBoxLayout(self.central_widget)
        self.video_display_label = QLabel("Video Display Area", self)
        self.layout.addWidget(self.video_display_label)
        self.setup_brightness_ui()

        self.setWindowTitle("Video Editor")
        self.setGeometry(300, 300, 800, 600)
        self.upload_button = QPushButton("Upload Video", self)
        self.upload_button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.upload_button)

    def setup_brightness_ui(self):
        if not hasattr(self, 'brightness_slider'):
            self.brightness_label = QLabel("Brightness: 50", self)
            self.layout.addWidget(self.brightness_label)
            self.brightness_slider = QSlider(Qt.Horizontal, self)
            self.brightness_slider.setRange(0, 100)
            self.brightness_slider.setValue(50)
            self.brightness_slider.valueChanged[int].connect(self.on_brightness_change)
            self.layout.addWidget(self.brightness_slider)
            self.next_button_brightness = QPushButton("Next: Contrast", self)
            self.next_button_brightness.clicked.connect(self.setup_contrast_ui)
            self.layout.addWidget(self.next_button_brightness)
        else:
            self.brightness_label.setVisible(True)
            self.brightness_slider.setVisible(True)
            self.next_button_brightness.setVisible(True)

    def setup_denoise_ui(self):
        print("Setting up denoise UI")  # Debug print
        self.clear_layout()

        # Denoising Method Selection
        self.denoise_method_label = QLabel("Denoising Method:", self)
        self.layout.addWidget(self.denoise_method_label)
        self.denoise_method_combo = QComboBox(self)
        self.denoise_method_combo.addItems(['gaussian', 'median', 'bilateral'])
        self.layout.addWidget(self.denoise_method_combo)

        # Denoising Parameter Slider
        self.denoise_param_label = QLabel("Denoising Parameter: 5", self)
        self.layout.addWidget(self.denoise_param_label)
        self.denoise_param_slider = QSlider(Qt.Horizontal, self)
        self.denoise_param_slider.setRange(1, 10)
        self.denoise_param_slider.setValue(5)
        self.denoise_param_slider.valueChanged[int].connect(self.on_denoise_param_change)
        self.layout.addWidget(self.denoise_param_slider)

        
        self.save_video_button = QPushButton("Save Video", self)
        self.save_video_button.clicked.connect(self.save_video)
        self.layout.addWidget(self.save_video_button)
        # Next or Finish Button
        self.next_button_denoise = QPushButton("Finish", self)
        self.next_button_denoise.clicked.connect(self.finish_editing)
        self.layout.addWidget(self.next_button_denoise)

    def setup_contrast_ui(self):
        self.brightness_label.setVisible(False)
        self.brightness_slider.setVisible(False)
        self.next_button_brightness.setVisible(False)
        self.upload_button.setVisible(False) 

        if not hasattr(self, 'contrast_slider'):
            self.contrast_label = QLabel("Contrast: 50", self)
            self.layout.addWidget(self.contrast_label)
            self.contrast_slider = QSlider(Qt.Horizontal, self)
            self.contrast_slider.setRange(0, 100)
            self.contrast_slider.setValue(50)
            self.contrast_slider.valueChanged[int].connect(self.on_contrast_change)
            self.layout.addWidget(self.contrast_slider)
            self.next_button_contrast = QPushButton("Next: Saturation", self)
            self.next_button_contrast.clicked.connect(self.setup_saturation_ui)
            self.layout.addWidget(self.next_button_contrast)
        else:
            self.contrast_label.setVisible(True)
            self.contrast_slider.setVisible(True)
            self.next_button_contrast.setVisible(True)

    def setup_saturation_ui(self):
        self.contrast_label.setVisible(False)
        self.contrast_slider.setVisible(False)
        self.next_button_contrast.setVisible(False)
        self.upload_button.setVisible(False)

        if not hasattr(self, 'saturation_slider'):
            self.saturation_label = QLabel("Saturation: 50", self)
            self.layout.addWidget(self.saturation_label)
            self.saturation_slider = QSlider(Qt.Horizontal, self)
            self.saturation_slider.setRange(0, 100)
            self.saturation_slider.setValue(50)
            self.saturation_slider.valueChanged[int].connect(self.on_saturation_change)
            self.layout.addWidget(self.saturation_slider)
        else:
            self.saturation_label.setVisible(True)
            self.saturation_slider.setVisible(True)

        self.next_button_saturation = QPushButton("Next: Denoise", self)
        self.next_button_saturation.clicked.connect(self.setup_denoise_ui)
        self.layout.addWidget(self.next_button_saturation)
        print("Next: Denoise button setup complete")  # Debug print


    def clear_layout(self):
        for i in reversed(range(self.layout.count())): 
            widget = self.layout.itemAt(i).widget()
            if widget is not None:
                widget.setVisible(False)

    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile() and url.toLocalFile().lower().endswith('.avi'):
                event.acceptProposedAction()
            else:
                print("The dragged file is not a local .avi file.")
        else:
            print("Drag event contains no URLs or is not a local file.")



    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            print(f"Dropped file path: {file_path}")
            self.load_video(file_path)
            event.acceptProposedAction()
        else:
            print("Drop event contains no URLs or is not a local file.")



    def on_brightness_change(self, value):
        self.brightness_label.setText(f"Brightness: {value}")
        self.brightness_value = value
        self.update_frame()


    def on_contrast_change(self, value):
        self.contrast_label.setText(f"Contrast: {value}")
        self.contrast_value = value
        self.update_frame()

    def on_saturation_change(self, value):
        self.saturation_label.setText(f"Saturation: {value}")
        self.saturation_value = value
        self.update_frame()

    def on_denoise_param_change(self, value):
        print(f"Denoising parameter changed to: {value}")  # Debug print
        self.denoise_param_label.setText(f"Denoising Parameter: {value}")
        try:
            self.update_frame()
        except Exception as e:
            print(f"Error updating frame: {e}")


    def adjust_brightness(self, frame, value):
        # Convert the value to a scale (you might need to adjust the scale factor)
        brightness_factor = (value - 50) / 25.0
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness_factor)
        return frame

    def adjust_contrast(self, frame, value):
        # Convert the value to a scale (you might need to adjust the scale factor)
        contrast_factor = 1 + (value - 50) / 50.0
        frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=0)
        return frame

    def adjust_saturation(self, frame, value):
        # Convert frame to HSV (hue, saturation, value)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Convert the value to a scale (you might need to adjust the scale factor)
        saturation_factor = (value - 50) / 50.0
        s = cv2.convertScaleAbs(s, alpha=saturation_factor, beta=0)

        # Merge back the channels and convert to BGR
        hsv = cv2.merge([h, s, v])
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame


    def load_video(self, file_path=None):
        print("Loading video...")  # Debug print

        # Release existing video capture if it exists
        if self.cap is not None:
            self.cap.release()

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi)")
        
        if file_path:
            print(f"Video path: {file_path}")  # Debug print
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)

            if self.cap.isOpened():
                print("Video capture opened successfully.")  # Debug print
                self.timer.start(30)
            else:
                print("Failed to open video capture.")  # Debug print
        else:
            print("No file selected.")  # Debug print


    def update_frame(self):
        if self.cap is None:
            print("self.cap is None")
        elif not self.cap.isOpened():
            print("self.cap is not opened")
        else:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                processed_frame = self.process_frame(frame)
                self.display_frame(processed_frame)
            else:
                print("Failed to read frame")

    def denoise(self, frame, method='gaussian', kernel_size=5):
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        if method == 'gaussian':
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        elif method == 'median':
            return cv2.medianBlur(frame, kernel_size)
        elif method == 'bilateral':
            return cv2.bilateralFilter(frame, kernel_size, 75, 75)
        else:
            raise NotImplementedError(f"denoise method {method} not understood")



    def process_frame(self, frame):
        try:
            if hasattr(self, 'denoise_param_slider'):
                denoise_method = self.denoise_method_combo.currentText()
                denoise_param = self.denoise_param_slider.value()
                frame = self.denoise(frame, method=denoise_method, kernel_size=denoise_param)
        
            # Apply brightness adjustment if brightness_slider exists
            if hasattr(self, 'brightness_slider'):
                brightness_value = self.brightness_slider.value()
                frame = self.adjust_brightness(frame, brightness_value)

            # Apply contrast adjustment if contrast_slider exists
            if hasattr(self, 'contrast_slider'):
                contrast_value = self.contrast_slider.value()
                frame = self.adjust_contrast(frame, contrast_value)

            # Apply saturation adjustment if saturation_slider exists
            if hasattr(self, 'saturation_slider'):
                saturation_value = self.saturation_slider.value()
                frame = self.adjust_saturation(frame, saturation_value)

            return frame
        except Exception as e:
            print(f"Error in process_frame: {e}")
            raise


    def display_frame(self, frame):
        # Convert frame to format suitable for PyQt and display
        qformat = QImage.Format_Indexed8 if len(frame.shape) == 2 else QImage.Format_RGB888
        out_image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        self.video_display_label.setPixmap(QPixmap.fromImage(out_image))

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.avi)")
        if file_path:
            self.load_video(file_path)

    def finish_editing(self):
        print("Editing finished")
        # You can add any finalization code here
    
    def save_video(self):
        if not self.video_path:
            print("No video loaded to save.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "AVI Files (*.avi)")
        if save_path:
            print(f"Saving video to: {save_path}")

            # Open the original video
            cap = cv2.VideoCapture(self.video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_path, fourcc, frame_rate, (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Apply adjustments to the frame
                    frame = self.adjust_brightness(frame, self.brightness_value)
                    frame = self.adjust_contrast(frame, self.contrast_value)
                    frame = self.adjust_saturation(frame, self.saturation_value)

                    # Write the frame
                    out.write(frame)
                else:
                    break

            # Release everything when done
            cap.release()
            out.release()
            print("Video saved successfully.")
        else:
            print("Save operation canceled.")


def main():
    app = QApplication(sys.argv)
    ex = VideoEditor()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()