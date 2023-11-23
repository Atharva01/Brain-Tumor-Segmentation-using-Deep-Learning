# src/pipeline.py
from src.data_processing import load_data, preprocess_data, augment_data
from src.model import unet_model
from src.train import train_model
from src.evaluate import evaluate_model

def main_pipeline():
    """
    Main pipeline for the Brain Tumor Segmentation project.
    """
    # Define paths to your training data
    train_data_paths = ['/path/to/train_data_1', '/path/to/train_data_2', ...]

    # Load and preprocess training data
    train_images, train_labels = load_data(train_data_paths)
    train_images, train_labels = preprocess_data(train_images, train_labels)
    train_images, train_labels = augment_data(train_images, train_labels)

    # Train the model
    train_model()

    # Define paths to your evaluation data
    eval_data_paths = ['/path/to/eval_data_1', '/path/to/eval_data_2', ...]

    # Load and preprocess evaluation data
    eval_images, eval_labels = load_data(eval_data_paths)
    eval_images, eval_labels = preprocess_data(eval_images, eval_labels)

    # Evaluate the model
    evaluate_model()

if __name__ == "__main__":
    main_pipeline()
