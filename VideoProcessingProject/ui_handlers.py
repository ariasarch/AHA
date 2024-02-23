import video_processing as vp

class VideoUIHandlers:
    def __init__(self, video_editor_ui):
        self.video_editor_ui = video_editor_ui

    def apply_denoise(self):
        if self.video_editor_ui.current_frame_index < len(self.video_editor_ui.video_frames):
            frame = self.video_editor_ui.video_frames[self.video_editor_ui.current_frame_index]
            kernel_size = self.video_editor_ui.denoise_slider.value()
            denoised_frame = vp.denoise(frame, method='gaussian', kernel_size=kernel_size)
            self.video_editor_ui.display_frame_from_array(denoised_frame)

    def estimate_motion(self):
        if self.video_editor_ui.current_frame_index > 0 and self.video_editor_ui.current_frame_index < len(self.video_editor_ui.video_frames):
            current_frame = self.video_editor_ui.video_frames[self.video_editor_ui.current_frame_index]
            previous_frame = self.video_editor_ui.video_frames[self.video_editor_ui.current_frame_index - 1]
            motion_vector = vp.estimate_motion(current_frame, previous_frame)
            # Process the motion_vector as needed

    def remove_background(self):
        if self.video_editor_ui.current_frame_index < len(self.video_editor_ui.video_frames):
            frame = self.video_editor_ui.video_frames[self.video_editor_ui.current_frame_index]
            bg_removed_frame = vp.remove_background(frame, method="uniform", kernel_size=5)
            self.video_editor_ui.display_frame_from_array(bg_removed_frame)

    def initialize_seeds(self):
        if self.video_editor_ui.current_frame_index < len(self.video_editor_ui.video_frames):
            frame = self.video_editor_ui.video_frames[self.video_editor_ui.current_frame_index]
            seeds = vp.seeds_init(frame, threshold=100, min_distance=10)
            # Process seeds as needed
            # For example, you might want to display these seeds on the UI
            # This could involve drawing circles or points on the frame at the seed locations
            # and then updating the display. Here's a simple way to do this:

            for seed in seeds:
                cv2.circle(frame, seed, radius=5, color=(0, 255, 0), thickness=-1)
            
            self.video_editor_ui.display_frame_from_array(frame)
            # Optionally, store the seeds for later use
            self.video_editor_ui.seeds = seeds

    def save_video(self):
        # Define the filename and format
        filename = "saved_video.avi"
        format = 'avi'

        # Check if there are frames to save
        if hasattr(self.video_editor_ui, 'video_frames') and self.video_editor_ui.video_frames:
            success = vp.save_minian(self.video_editor_ui.video_frames, filename, format)
            if success:
                print("Video saved successfully as", filename)
                self.video_editor_ui.statusBar().showMessage("Video saved successfully")
            else:
                print("Failed to save video")
                self.video_editor_ui.statusBar().showMessage("Failed to save video")

    def update_background(self):
        if self.video_editor_ui.current_frame_index < len(self.video_editor_ui.video_frames):
            frame = self.video_editor_ui.video_frames[self.video_editor_ui.current_frame_index]
            # Assuming update_background_method is a function in your video_processing module
            updated_background = vp.update_background_method(frame)
            # Now you can do something with the updated background
            # For example, you might want to display it in the UI or store it
            self.video_editor_ui.display_frame_from_array(updated_background)
            self.video_editor_ui.updated_background = updated_background


    def update_temporal(self):
        # Assuming you have a list of frames and temporal components to update
        if hasattr(self.video_editor_ui, 'temporal_components'):
            updated_temporal_components = vp.update_temporal_method(self.video_editor_ui.video_frames, 
                                                                    self.video_editor_ui.temporal_components)
            # Store the updated temporal components
            self.video_editor_ui.temporal_components = updated_temporal_components
            # Update the UI or perform further processing as needed


    def generate_videos(self):
        transformations = [vp.apply_transform]  # List of transformation functions

        if self.video_editor_ui.current_frame_index > 0 and self.video_editor_ui.current_frame_index < len(self.video_editor_ui.video_frames):
            current_frame = self.video_editor_ui.video_frames[self.video_editor_ui.current_frame_index]
            previous_frame = self.video_editor_ui.video_frames[self.video_editor_ui.current_frame_index - 1]
            motion_vector = vp.estimate_motion(current_frame, previous_frame)

            # Applying the transformation to each frame
            generated_videos = [transform(current_frame, motion_vector) for transform in transformations]

            # Handle the generated videos
            # Example: Display the first transformed video frame in the UI
            self.video_editor_ui.display_frame_from_array(generated_videos[0])

            # If you want to save these frames as a video, you can use vp.save_minian or similar function
            # vp.save_minian(generated_videos, "transformed_video.avi", format='avi')


    def save_minian(self):
        data_to_save = self.video_editor_ui.processed_data  # Example data to save
        filename = "output.minian"  # Example filename
        format = 'avi'  # Example format
        success = vp.save_minian(data_to_save, filename, format)
        if success:
            # Display a message or update the UI to indicate the data has been saved
            self.video_editor_ui.statusBar().showMessage("Data saved successfully")
        else:
            # Handle save failure
            self.video_editor_ui.statusBar().showMessage("Failed to save data")

