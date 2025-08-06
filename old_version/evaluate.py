# src/evaluate.py
import tensorflow as tf
from src.data_processing import load_data, preprocess_data
from src.model import unet_model

def evaluate_model():
    """
    Evaluate the U-Net model.
    """
    # Define paths to your evaluation data
    eval_data_paths = ['/path/to/eval_data_1', '/path/to/eval_data_2', ...]

    # Load and preprocess evaluation data
    eval_images, eval_labels = load_data(eval_data_paths)
    eval_images, eval_labels = preprocess_data(eval_images, eval_labels)

    # Define the U-Net model
    model = unet_model(input_shape=(128, 128, 1))

    # Load the trained weights
    model.load_weights("models/model.h5")

    # Compile the model (this is necessary even for evaluation)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Evaluate the model
    evaluation_results = model.evaluate(eval_images, eval_labels, verbose=1)

    # Display the evaluation results
    print(f"Accuracy: {evaluation_results[1] * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model()
