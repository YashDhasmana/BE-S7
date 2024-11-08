import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Step 1: Load and Preprocess the Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), mode='constant')  # Padding to 32x32
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), mode='constant')
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0

# One-hot encoding the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Step 2: Define the LeNet Model
def create_lenet_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(32, 32, 1)))  # Input layer
    model.add(layers.Conv2D(6, kernel_size=(5, 5), activation='sigmoid'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='sigmoid'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    # Reshape the output for the dense layers
    model.add(layers.Reshape((5 * 5 * 16,)))  # Reshape to (400,) before dense layers
    model.add(layers.Dense(120, activation='sigmoid'))  # First dense layer with 120 units
    model.add(layers.Dense(84, activation='sigmoid'))   # Second dense layer with 84 units
    model.add(layers.Dense(10, activation='softmax'))    # Output layer for 10 classes

    return model

model = create_lenet_model()

# Step 3: Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

# Step 5: Evaluate the Model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')
