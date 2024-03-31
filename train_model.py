import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import os

# Set the paths to your dataset directories
train_dir = '/Users/yashaggarwal/Desktop/archive/train'
test_dir = '/Users/yashaggarwal/Desktop/archive/test'

# Set the image size and batch size
img_size = (32, 32)
batch_size = 64

# Define data transforms
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create datasets and data loaders
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize the model, loss function, and optimizer
model = CNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
    for images, labels in progress_bar:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/len(train_loader))



# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.float().unsqueeze(1)).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')



# Function to predict if an image is AI-generated or not
def predict_image(image_path):
    image = transform(datasets.folder.default_loader(image_path)).unsqueeze(0)
    output = model(image)
    predicted = (output >= 0.5).float().item()
    return 'AI-generated' if predicted else 'Real'

# Example usage
image_path = '/Users/yashaggarwal/Desktop/archive/test/1a.jpg'
prediction = predict_image(image_path)
print(f'The image is predicted to be: {prediction}')

# Example usage
image_path = '/Users/yashaggarwal/Desktop/archive/test/1b.jpg'
prediction = predict_image(image_path)
print(f'The image is predicted to be: {prediction}')

# Save the trained model
model_path = '/Users/yashaggarwal/Desktop/archive/saved_model'
torch.save(model.state_dict(), model_path)