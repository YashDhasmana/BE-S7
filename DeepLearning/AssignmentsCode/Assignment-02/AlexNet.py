import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

# Step 1: Load the ImageNet Dataset
def load_imagenet_data(batch_size=32):
    # Load the dataset and resize images to 227x227
    dataset, info = tfds.load('imagenet2012', split=['train', 'validation'], as_supervised=True, with_info=True)

    def preprocess(image, label):
        image = tf.image.resize(image, [227, 227])
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
        return image, label

    train_dataset = dataset[0].map(preprocess).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = dataset[1].map(preprocess).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset

# Step 2: Define the Original AlexNet Model
def create_alexnet_model():
    model = models.Sequential()

    # First Convolutional Layer
    model.add(layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Second Convolutional Layer
    model.add(layers.Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Third Convolutional Layer
    model.add(layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))

    # Fourth Convolutional Layer
    model.add(layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))

    # Fifth Convolutional Layer
    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Flattening the output
    model.add(layers.Flatten())

    # Fully Connected Layers
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(1000, activation='softmax'))  # Output layer for 1000 classes

    return model

# Step 3: Load Data
train_dataset, val_dataset = load_imagenet_data(batch_size=32)

# Step 4: Compile the Model
model = create_alexnet_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the Model
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Step 6: Evaluate the Model (you can evaluate on a separate test set if needed)
# test_loss, test_accuracy = model.evaluate(test_dataset)
# print(f'Test accuracy: {test_accuracy:.4f}')
