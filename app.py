from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import subprocess

app = Flask(__name__)

# Configuration
# app.config['SECRET_KEY'] = 'your_secret_key'  # Needed for session management and flash messaging
# app.config['videos'] = 'uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# # Ensure the upload directory exists
# os.makedirs(app.config['videos'], exist_ok=True)

# # Dummy function to process video and return PP score
# def process_video(video_path):
#     # Assuming 'video_path' is the path to the uploaded video file
#     # And assuming your scripts work with the full path to a specific video file rather than a directory of videos
#     # If not, you might need to adjust how you pass arguments to your scripts

#     # Path to the Python interpreter, use 'python' or 'python3' depending on your environment
#     python_executable = 'python3'

#     # The path to your scripts
#     script1 = 'path/to/vtf.py'  # Adjust the path as necessary
#     script2 = 'path/to/predict.py'  # Adjust the path as necessary

#     # The directory where the frames will be saved (and read from by the second script)
#     frames_directory = 'path/to/frames'  # Adjust the path as necessary

#     # Run the first script, assuming it prepares the video at 'video_path' and saves frames to 'frames_directory'
#     subprocess.run([python_executable, script1, video_path, frames_directory])

#     # Run the second script and capture its output, assuming it reads frames from 'frames_directory' and prints pp_score
#     result = subprocess.run([python_executable, script2, frames_directory], capture_output=True, text=True)

#     # Extract pp_score from the output
#     pp_score = result.stdout.strip()  # This assumes the output is just the pp_score or can be directly used

#     return pp_score

@app.route('/')
def home():
    pp_score = 34
    return render_template('AI-output.html', pp_score=pp_score)

if __name__ == '__main__':
    app.run(debug=True)
