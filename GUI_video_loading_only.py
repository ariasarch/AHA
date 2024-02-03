import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        self.select_button = QPushButton('Select AVI File', self)
        self.select_button.clicked.connect(self.open_avi_file)
        self.layout.addWidget(self.select_button)

        self.setLayout(self.layout)

    def open_avi_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open AVI File', '', 'AVI files (*.avi)')
        if file_path:
            self.play_video(file_path)

    def play_video(self, file_path):
        cap = cv2.VideoCapture(file_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape

            # Scale the frame to fit in the label widget
            max_size = max(h, w)
            if max_size > 800:
                scale_factor = 800 / max_size
                rgb_frame = cv2.resize(rgb_frame, (int(w * scale_factor), int(h * scale_factor)))

            # Convert the frame to QPixmap and set it as the label's pixmap
            qimg = QPixmap.fromImage(QImage(rgb_frame.data, w, h, QImage.Format_RGB888))
            self.label.setPixmap(qimg)

            # Display the frame for 30 milliseconds
            cv2.waitKey(30)  

        cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())