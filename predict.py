import torch
from torchvision import transforms, datasets
from model import CNN
import os

# Set the path to the trained model
model_path = '/Users/yashaggarwal/Desktop/archive/saved_model'
# Load the trained model
model = CNN()
model.load_state_dict(torch.load(model_path))
model.eval()

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
folder_path = '/Users/yashaggarwal/Desktop/Catapult/fake_images'

# Get a list of all image file names in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

# Iterate over each image file and predict if it is AI-generated or real
for image_file in image_files:
    count += 1
    image_path = os.path.join(folder_path, image_file)
    prediction = predict_image(image_path)
    print(f'Image: {image_file}')
    print(f'Prediction: {prediction}')
    print('------------------------')

    print("count = ", count)
    print("real = ", real)