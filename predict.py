import torch
from torchvision import transforms, datasets
from model import CNN
import os

import random
import string

def generate_wpa_key(length):
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))
  # 504 bits = 63 bytes

# Set the path to the trained model
model_path = '/Users/yashaggarwal/Desktop/archive/saved_model'
# Load the trained model
model = CNN()
model.load_state_dict(torch.load(model_path))
model.eval()

def CreateFile(x, key):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pixel Proof - Video Uploader</title>
    <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
    <style>
        body {
        margin: 0;
        background: #000;
        font-family: 'Work Sans', sans-serif;
        font-weight: 800;
        color: white;
        }

        .container {
        width: 80%;
        margin: 0 auto;
        }

        header {
        background: #000;
        margin-bottom: 20px;
        overflow: hidden;
        }

        header::after {
        content: '';
        display: table;
        clear: both;
        }

        .logo {
        float: left;
        padding: 10px 0;
        font-size: 1.5rem;
        font-weight: bold;
        color: #55d6aa;
        text-decoration: none;
        margin-right: 40px;
        font-family: 'Press Start 2P', cursive;
        }

        nav {
        float: right;
        }

        nav ul {
        margin: 0;
        padding: 0;
        list-style: none;
        }

        nav li {
        display: inline-block;
        margin-left: 70px;
        padding-top: 23px;
        position: relative;
        }

        nav a {
        color: white;
        text-decoration: none;
        text-transform: uppercase;
        font-size: 14px;
        }

        nav a:hover {
        color: #55d6aa;
        }

        nav a::before {
        content: '';
        display: block;
        height: 5px;
        background-color: #55d6aa;
        position: absolute;
        top: 0;
        width: 0%;
        transition: all ease-in-out 250ms;
        }

        nav a:hover::before {
        width: 100%;
        }

        .navbar {
        background: none;
        padding: 0;
        border: none;
        }

        .navbar-logo {
        color: #55d6aa;
        text-decoration: none;
        }

        .navbar-links {
        margin: 0;
        padding: 0;
        }

        .navbar-links li {
        display: inline-block;
        margin-left: 40px;
        }

        .navbar-links a {
        color: white;
        text-decoration: none;
        font-size: 14px;
        }

        .navbar-links a:hover {
        color: #55d6aa;
        }

        .content {
        background-color: rgba(0, 0, 0, 0.5);
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        max-width: 400px;
        width: 100%;
        margin: 3rem auto; /* Center the content vertically and horizontally */
        }

        h1 {
        color: white;
        font-size: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        margin-bottom: 1rem; /* Add margin to separate heading from buttons */
        }

        .button {
        color: white;
        background-color: rgba(0, 0, 0, 0.5);
        padding: 10px 20px;
        border: 1px solid white;
        border-radius: 4px;
        margin: 10px;
        text-decoration: none;
        transition: background-color 0.3s, color 0.3s, transform 0.3s;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.5);
        }

        .button:hover {
        background-color: white;
        color: black;
        transform: translateY(-4px);
        }

        .video-container {
        margin-top: 2rem;
        text-align: center; /* Center the video container */
        }

        video {
        width: 100%;
        max-width: 800px;
        height: auto;
        border-radius: 10px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
        }

        .popup {
        background-color: rgba(0, 0, 0, 0.5);
        color: white;
        padding: 20px;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        display: none;
        z-index: 999;
        justify-content: center;
        align-items: center;
        }

        .popup-box {
        background-color: #000;
        padding: 80px; /* Adjusted padding to make the box larger */
        border-radius: 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3), 0 0 40px 5px rgba(255, 255, 255, 0.5); /* Added white highlight */
        max-width: 400px;
        width: 100%;
        text-align: center;
        }

        .loader {
        animation: rotate 1s infinite;
        height: 100px;
        width: 100px;
        margin: auto;
        }

        .loader:before,
        .loader:after {
        border-radius: 50%;
        content: '';
        display: block;
        height: 20px;
        width: 20px;
        }

        .loader:before {
        animation: ball1 1s infinite;
        background-color: #cb2025;
        box-shadow: 30px 0 0 #f8b334;
        margin-bottom: 10px;
        }

        .loader:after {
        animation: ball2 1s infinite;
        background-color: #00a096;
        box-shadow: 30px 0 0 #97bf0d;
        }

        @keyframes rotate {
        0% {
            transform: rotate(0deg) scale(0.8);
        }
        50% {
            transform: rotate(360deg) scale(1.2);
        }
        100% {
            transform: rotate(720deg) scale(0.8);
        }
        }

        @keyframes ball1 {
        0% {
            box-shadow: 30px 0 0 #f8b334;
        }
        50% {
            box-shadow: 0 0 0 #f8b334;
            margin-bottom: 0;
            transform: translate(15px, 15px);
        }
        100% {
            box-shadow: 30px 0 0 #f8b334;
            margin-bottom: 10px;
        }
        }

        @keyframes ball2 {
        0% {
            box-shadow: 30px 0 0 #97bf0d;
        }
        50% {
            box-shadow: 0 0 0 #97bf0d;
            transform: translate(15px, 15px);
        }
        100% {
            box-shadow: 30px 0 0 #97bf0d;
        }
        }

        .confidence-level-box {
        text-align: center;
        margin-bottom: 20px;
        }

        .button {
        display: block;
        margin: 0 auto;
        }

        .container {
        width: 100%;
        }

        header {
        background: #000;
        margin-bottom: 30px;
        overflow: hidden;
        }

        header::after {
        content: '';
        display: table;
        clear: both;
        }

        .logo {
        float: left;
        padding-top: 9px;
        font-size: 1.5rem;
        font-weight: bold;
        color: #55d6aa;
        text-decoration: none;
        margin-right: 40px;
        font-family: 'Press Start 2P', cursive;
        }

        nav {
        float: right;
        }

        nav ul {
        margin: 0;
        padding: 0;
        list-style: none;
        }

        nav li {
        display: inline-block;
        margin-left: 70px;
        padding-top: 10px;
        position: relative;
        }

        nav a {
        color: white;
        text-decoration: none;
        text-transform: uppercase;
        font-size: 14px;
        }

        nav a:hover {
        color: #55d6aa;
        }

        nav a::before {
        content: '';
        display: block;
        height: 5px;
        background-color: #55d6aa;
        position: absolute;
        top: 0;
        width: 0%;
        transition: all ease-in-out 250ms;
        }

        nav a:hover::before {
        width: 100%;
        }

        .navbar {
        background: none;
        padding: 0;
        border: none;
        }

        .navbar-logo {
        color: #55d6aa;
        text-decoration: none;
        }

        .navbar-links {
        margin: 0;
        padding: 0;
        }

        .navbar-links li {
        display: inline-block;
        margin-left: 40px;
        }

        .navbar-links a {
        color: white;
        text-decoration: none;
        font-size: 14px;
        }

        .navbar-links a:hover {
        color: #55d6aa;
        }
    </style>
    </head>
    <body>
    <header>
        <div class="container">
        <a href="home.html" class="logo navbar-logo">Pixel Proof</a>
        <nav>
            <ul class="navbar-links">
            <li><a href="home.html" target="_blank">Home</a></li>
            <li><a href="#" target="_blank">About Us</a></li>
            <li><a href="#" target="_blank">Contact</a></li>
            </ul>
        </nav>
        </div>
    </header>

    <div class="container">
        <div class="content">
        <video id="uploadedVideo" controls></video>
        </div>

        <div class="confidence-level-box">
        <p class="real-or-ai" style="font-size: 30px; color: #55d6aa;">PP Key: """ + key + """</p>
        <p class="real-or-ai" style="font-size: 30px; color: #55d6aa;">PP Score: """ + str(x)+ """</p>
        </div>

        <a href = "http://127.0.0.1:5501/index2.html"><button class="button">Back</button></a>
    </div>

    <script>
        const uploadedVideo = document.getElementById('uploadedVideo');
        uploadedVideo.src = 'videos/clip.mp4';
    </script>
    </body>
    </html>
    """


    file_path = "AI-output.html"
    with open(file_path, "w") as file:
        file.write(html_content)


# Define the data transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

count = 0
real = 0 

# Function to predict if an image is AI-generated or not with confidence percentage
def predict_image(image_path):
    global real
    image = transform(datasets.folder.default_loader(image_path)).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        confidence = output.item()
        predicted = 'AI-generated' if confidence >= 0.5 else 'Real'
        if (predicted == 'Real'):
            real += 1
        confidence_percentage = confidence * 100 if predicted == 'AI-generated' else (1 - confidence) * 100
        return f'{predicted} with {confidence_percentage:.2f}% confidence'

# Set the path to the folder containing the images
folder_path = '/Users/yashaggarwal/Desktop/Catapult/saved_frames'

# Get a list of all image file names in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

# Iterate over each image file and predict if it is AI-generated or real
for image_file in image_files:
    count += 1
    image_path = os.path.join(folder_path, image_file)
    prediction = predict_image(image_path)
    #print(f'Image: {image_file}')
    #print(f'Prediction: {prediction}')
    #print('------------------------')

    #print("count = ", count)
    #print("real = ", real)
result = ((real / count) * 100 + 50) if ((real / count) * 100 < 50) else ((real / count) * 100)
wpa_key = generate_wpa_key(63)
CreateFile(result,wpa_key)