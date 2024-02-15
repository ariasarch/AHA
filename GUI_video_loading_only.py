import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading
import matplotlib
matplotlib.use("Qt5Agg")

import xarray as xr

video_path = 'c:\\Users\\heather\\Desktop\\GUI\\Sample_avi_clip.avi'
cap = cv2.VideoCapture(video_path)

# Get video metadata
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_channels = 3  # Assuming RGB video, change if necessary


data_fr=np.empty((frame_count, frame_height, frame_width, num_channels))

# Create an empty xarray with dimensions time, y, x, and channel
xarr = xr.DataArray(data_fr,
                    dims=['time', 'y', 'x', 'channel'],
                    coords={'time': np.arange(frame_count) / fps})

# Convert each frame to xarray
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV captures the frame in BGR format, so we convert it to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    xarr[i] = frame_rgb



# Release the video capture and close the video file
cap.release()

class VideoPlayer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 800, 600)

        # Create a Matplotlib Figure and Canvas
        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        self.setCentralWidget(self.canvas)

        # Create the subplot for the video display
        self.ax = self.fig.add_subplot(111)

        # Set up animation timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Keep track of the current frame
        self.current_frame = 0

    def play(self):
        self.timer.start(100)  # Set the timer interval (milliseconds)
    
    def stop(self):
        self.timer.stop()

    def update_frame(self):
        self.current_frame = (self.current_frame + 1) % len(xarr.time)

        # Get the current frame data
        frame = xarr[self.current_frame].values

        # Display the frame using imshow
        self.ax.imshow(frame)
        self.canvas.draw()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    player.play()
    sys.exit(app.exec_())