from data_preprocessing import load_and_preprocess_data
from model import unet_model
from train import train_model
from evaluate import evaluate_model

def main_pipeline():
    """
    Main pipeline for the Brain Tumor Segmentation project.
    """
    # Define the directory containing the BraTS files
    data_dir = '.'

    # Load and preprocess data using data_preprocessing.py
    train_images, train_labels, eval_images, eval_labels = load_and_preprocess_data(data_dir)

    # Train the model
    train_model(train_images, train_labels)

    # Evaluate the model
    evaluate_model(eval_images, eval_labels)

if __name__ == "__main__":
    main_pipeline()
