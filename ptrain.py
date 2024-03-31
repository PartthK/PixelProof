import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from model import CNN
import os

# Set the path to the trained model
model_path = '/Users/yashaggarwal/Desktop/archive/saved_model'

# Load the trained model
model = CNN()
model.load_state_dict(torch.load(model_path))
model.train()  # Set the model to training mode

# Define the data transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Set the path to the folder containing the images
folder_path = '/Users/yashaggarwal/Desktop/Catapult/images'

# Get a list of all image file names in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set the number of epochs for self-supervised learning
epochs = 10

# Perform self-supervised learning
for epoch in range(epochs):
    running_loss = 0.0
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = transform(datasets.folder.default_loader(image_path)).unsqueeze(0)
        
        # Forward pass
        output = model(image)
        
        # Calculate loss assuming all images are AI-generated
        target = torch.tensor([[0.0]])  # 0.0 represents AI-generated
        loss = criterion(output, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(image_files):.4f}')

# Save the improved model
new_model_path = '/Users/yashaggarwal/Desktop/archive/new_model'
torch.save(model.state_dict(), new_model_path)

print('Self-supervised learning completed. Improved model saved.')