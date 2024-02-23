 # Entry point of the application
import sys
from PyQt5.QtWidgets import QApplication
from main_window import VideoEditorUI

def main():
    # Create the application instance
    app = QApplication(sys.argv)

    # Create an instance of the VideoEditorUI
    video_editor = VideoEditorUI()

    # Show the UI
    video_editor.show()

    # Execute the application's main loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
