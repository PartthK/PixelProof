import cv2
import os

# Define the folder where you want to save the images
folder_path = 'saved_frames'

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Open the video file
video = cv2.VideoCapture('videos/clip.mp4')

# Initialize a counter for the frames
frame_count = 0

# Maximum number of frames to save
max_frames = 1000

# Loop through the frames
while True:
    # Read a frame from the video
    ret, frame = video.read()

    # If there are no more frames or we've reached the maximum, break out of the loop
    if not ret or frame_count == max_frames:
        break

    # Save the frame as an image file
    cv2.imwrite(os.path.join(folder_path, f'frame_{frame_count}.jpg'), frame)

    # Increment the frame counter
    frame_count += 1

# Release the video capture object
video.release()

print(f'Saved {frame_count} frames in "{folder_path}" folder.')
