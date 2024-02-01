import tkinter as tk
from tkinter import ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
from PIL import Image, ImageTk

# Constants for window size and layout configuration
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600
FILE_FRAME_ROW = 0
SIDEBAR_COLUMN = 0
FEATURES_COLUMN = 2
MAIN_CONTENT_COLUMN = 1
MAIN_CONTENT_ROW_START = 1
MAIN_CONTENT_ROWSPAN = 5
FEATURES_ROW_START = 1
FEATURES_ROWSPAN = 5  # Span equal to MAIN_CONTENT_ROWSPAN for symmetry
SIDE_COLUMN_WEIGHT = 1
MAIN_COLUMN_WEIGHT = 200  # Increase this to make the middle column bigger

# Colors for each division
SIDEBAR_COLOR = '#96C66E'    # Example light orange
FEATURES_COLOR = '#B670C8'   # Example light green

root = TkinterDnD.Tk()
root.title('Video Loader')
root.geometry(f'{WINDOW_WIDTH}x{WINDOW_HEIGHT}')

# Configure grid layout
for i in range(4):  # Assuming 4 rows including the file frame
    if i == FILE_FRAME_ROW:  # If it's the row of the file frame
        root.grid_rowconfigure(i, weight=1)  # Assign a smaller weight for less stretchability
    else:
        root.grid_rowconfigure(i, weight=3)  # Other rows get a larger weight
# Configure grid layout
for i in range(3):  # Assuming 3 columns
    if i == SIDEBAR_COLUMN or i == FEATURES_COLUMN:
        root.grid_columnconfigure(i, weight=SIDE_COLUMN_WEIGHT)
    else:
        root.grid_columnconfigure(i, weight=MAIN_COLUMN_WEIGHT)

# Style configuration for ttk elements
style = ttk.Style(root)
style.configure('TButton', relief='flat')
style.map('TButton', background=[('active', '#dddddd'), ('pressed', '#aaaaaa')])

# File Frame (top row of the grid, spanning all columns)
file_frame = tk.Frame(root)
file_frame.grid(row=FILE_FRAME_ROW, columnspan=3, sticky='nsew')  # Make sure this spans all columns

# Tab buttons
tabs = ['File', 'Project', 'Edit']
for tab in tabs:
    button = ttk.Button(file_frame, text=tab, style='TButton', command=lambda t=tab: print(f"{t} tab clicked"))
    button.pack(side=tk.LEFT)

# Sidebar Frame (smaller side column)
sidebar = tk.Frame(root, bg=SIDEBAR_COLOR)
sidebar.grid(row=MAIN_CONTENT_ROW_START, column=SIDEBAR_COLUMN, rowspan=FEATURES_ROWSPAN, sticky='nsew')

# Labels in Sidebar
labels = ["Project Name:", "Exp Name:", "Directory:"]
for i, label in enumerate(labels):
    tk.Label(sidebar, text=label, bg=SIDEBAR_COLOR).grid(row=i*2, column=0, sticky='nsew')
    tk.Label(sidebar, text="", bg=SIDEBAR_COLOR).grid(row=i*2+1, column=0, sticky='nsew')

# Features Frame (smaller side column)
features_frame = tk.Frame(root, bg=FEATURES_COLOR)
features_frame.grid(row=FEATURES_ROW_START, column=FEATURES_COLUMN, rowspan=FEATURES_ROWSPAN, sticky='nsew')

# Feature checkboxes
features = ["Speed", "Velocity", "Anxiety", "Stride"]
tk.Label(features_frame, text="Features", bg=FEATURES_COLOR).grid(row=0, sticky='nsew')

for i, feature in enumerate(features):
    # Create the Checkbutton with a command
    checkbutton = tk.Checkbutton(features_frame, text=feature, bg=FEATURES_COLOR, 
                                 command=lambda f=feature: print(f"{f} clicked"))

    # Grid the Checkbutton
    checkbutton.grid(row=i+1, sticky='nsew')

# Main Frame (larger middle column, adjust columnspan if needed)
main_frame = tk.Frame(root)
main_frame.grid(row=MAIN_CONTENT_ROW_START, column=MAIN_CONTENT_COLUMN, rowspan=MAIN_CONTENT_ROWSPAN, columnspan=1, sticky='nsew')  # Increase columnspan for a wider middle column

# Create frames within main_frame
small_frames = [[None for _ in range(2)] for _ in range(2)]

# Create frames within main_frame
for i in range(2):  # two rows
    for j in range(2):  # two columns
        small_frame = tk.Frame(main_frame, borderwidth=1, relief="solid")
        small_frame.grid(row=i, column=j, sticky='nsew', padx=5, pady=5)
        
        # Set a minimum size for each small frame
        small_frame.grid_propagate(False)
        small_frame.config(width=WINDOW_WIDTH-((6/10)*WINDOW_WIDTH), height=WINDOW_HEIGHT-((50/100)*WINDOW_HEIGHT))  # Adjust the size as needed

        # Store the frame reference
        small_frames[i][j] = small_frame

        # Configure weights
        main_frame.grid_rowconfigure(i, weight=1)
        main_frame.grid_columnconfigure(j, weight=1)

drag_and_drop_labels = []

# Messages for each frame
frame_messages = ["Drag and Drop Video Here", "Processed Video Here", "Plot One", "Plot Two"]

# Inside the loop where frames are created:
for i in range(2):  # Two rows
    for j in range(2):  # Two columns (since you have 4 frames)
        message = frame_messages[i * 2 + j]
        label = tk.Label(small_frames[i][j], text=message)
        label.grid(row=0, column=0, sticky='nsew')  # Use sticky option to center the text
        drag_and_drop_labels.append(label)  # Store the label for later reference

# Configure the rows and columns in the main_frame
for i in range(2):
    main_frame.grid_rowconfigure(i, weight=1, minsize=200)  # Set a minimum size

for j in range(2):
    main_frame.grid_columnconfigure(j, weight=1, minsize=200)  # Set a minimum size

# Now, create the label in the first small frame for video display
frame_1_label = tk.Label(small_frames[0][0])
frame_1_label.pack()

# Function to handle file drops
def handle_file_drop(event):
    video_path = event.data
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert the color format from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            img = Image.fromarray(frame)
            # Resize image to fit the label
            img = img.resize((small_frames[0][0].winfo_width(), small_frames[0][0].winfo_height()), Image.Resampling.LANCZOS)
            # Convert to ImageTk format
            imgtk = ImageTk.PhotoImage(image=img)
            frame_1_label.config(image=imgtk)
            frame_1_label.image = imgtk  # Keep a reference
            drag_and_drop_labels[0].pack_forget()  # This hides the label

        cap.release()

# Register the label as a drop target
small_frames[0][0].drop_target_register(DND_FILES)
small_frames[0][0].dnd_bind('<<Drop>>', handle_file_drop)

root.mainloop()

# add text box to project title and experiment name 
# add file explorer https://www.geeksforgeeks.org/file-explorer-in-python-using-tkinter/
# drop down menu for file 