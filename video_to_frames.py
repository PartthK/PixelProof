import cv2

# Open the video file
video = cv2.VideoCapture('videos/clip.mp4')

# Create a list to store the frames
frames = []

# Loop through the frames
while True:
    # Read a frame from the video
    ret, frame = video.read()

    # If there are no more frames, break out of the loop
    if not ret:
        break

    # Append the frame to the list
    frames.append(frame)

# Release the video capture object
video.release()

# At this point, the 'frames' list contains all the frames of the video
# You can process or save the frames as needed
for i, frame in enumerate(frames):
    # Save the frame as an image file (TODO change the file path to save to a folder!!!)
    cv2.imwrite(f'frame_{i}.jpg', frame)

print(f'Total frames: {len(frames)}')
print(f'Type: {type(frames)}')