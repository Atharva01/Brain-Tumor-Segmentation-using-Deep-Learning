# src/train.py
import datetime
import tensorflow as tf
from src.data_processing import load_data, preprocess_data, augment_data
from src.model import unet_model

def train_model():
    """
    Train the U-Net model.
    """
    # Define paths to your training data
    train_data_paths = ['/path/to/train_data_1', '/path/to/train_data_2', ...]

    # Load and preprocess training data
    train_images, train_labels = load_data(train_data_paths)
    train_images, train_labels = preprocess_data(train_images, train_labels)
    train_images, train_labels = augment_data(train_images, train_labels)

    # Define the U-Net model
    model = unet_model(input_shape=(128, 128, 1))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Set up TensorBoard callback for visualization
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    model.fit(train_images, train_labels, validation_split=0.1, batch_size=16, epochs=10, callbacks=[tensorboard_callback])

    # Save the trained model
    model.save_weights("models/model_for_tumor.h5")
    model_json = model.to_json()
    with open("models/model_for_tumor.json", "w") as json_file:
        json_file.write(model_json)

if __name__ == "__main__":
    train_model()
