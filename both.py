import subprocess
import os
import time

# Path to the file to delete

# Path to the Python interpreter, use 'python' or 'python3' depending on your environment
python_executable = 'python'

# The path to your scripts
script1 = 'vtf.py'
script2 = 'predict.py'

# The directory containing your video files
video_directory = 'path/to/videos'
# The directory where the frames will be saved (and read from by the second script)
frames_directory = 'path/to/frames'

# Run the first script
subprocess.run([python_executable, script1, video_directory, frames_directory])

# Run the second script
subprocess.run([python_executable, script2, frames_directory])

# Wait for 10 seconds before deleting the file
time.sleep(15)
file_to_delete = 'videos/clip.mp4'
# Delete the file
try:
    os.remove(file_to_delete)
    print(f"File '{file_to_delete}' deleted successfully.")
except FileNotFoundError:
    print(f"File '{file_to_delete}' not found.")
except PermissionError:
    print(f"No permission to delete '{file_to_delete}'.")
except Exception as e:
    print(f"An error occurred: {e}")
