import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set the paths to the fake and real image folders
fake_path = '/Users/yashaggarwal/Desktop/Catapult/fake'
real_path = '/Users/yashaggarwal/Desktop/Catapult/real'

# Set the image size and batch size
img_size = (224, 224)
batch_size = 32

# Create lists to store the image paths and labels
fake_images = []
real_images = []

# Load the fake images
for filename in os.listdir(fake_path):
    fake_images.append(os.path.join(fake_path, filename))

# Load the real images
for filename in os.listdir(real_path):
    real_images.append(os.path.join(real_path, filename))

# Create labels for fake and real images
fake_labels = ['fake'] * len(fake_images)
real_labels = ['real'] * len(real_images)

# Combine the image paths and labels into a DataFrame
data = pd.DataFrame({
    'filename': fake_images + real_images,
    'label': fake_labels + real_labels
})

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col='filename',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col='filename',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary')

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 10
steps_per_epoch = len(train_data) // batch_size
validation_steps = len(test_data) // batch_size

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_progress_bar = tqdm(train_generator, total=steps_per_epoch, desc='Training')
    for batch in train_progress_bar:
        model.train_on_batch(batch[0], batch[1])

    test_progress_bar = tqdm(test_generator, total=validation_steps, desc='Validation')
    for batch in test_progress_bar:
        loss, accuracy = model.test_on_batch(batch[0], batch[1])
        test_progress_bar.set_postfix(loss=loss, accuracy=accuracy)

# Save the model
model.save('/Users/yashaggarwal/Desktop/archive/final_model')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=validation_steps)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions on the test set
test_generator.reset()
predictions = model.predict(test_generator, steps=validation_steps)
predicted_classes = np.round(predictions).flatten()

# Print the predictions with confidence levels
for i, row in test_data.iterrows():
    predicted_class = predicted_classes[i]
    confidence = predictions[i][0]
    if predicted_class == 0:
        print(f"Image: {row['filename']}, Predicted: Fake, Confidence: {confidence:.4f}")
    else:
        print(f"Image: {row['filename']}, Predicted: Real, Confidence: {confidence:.4f}")