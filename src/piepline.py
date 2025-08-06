# pipeline.py
import pickle
import pandas as pd
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualization import export_tiff_predictions, export_ground_truth

if __name__ == "__main__":
    # Load partition info and metadata
    partition = pickle.load(open("./channel_split/partition.pkl", "rb"))
    survival_data = pd.read_csv('survival_data.csv')

    # Build tumor type dictionary
    import os
    HGG_dir_list = next(os.walk('./HGG/'))[1]
    LGG_dir_list = next(os.walk('./LGG/'))[1]
    tumor_type_dict = {}
    for patientID in HGG_dir_list + LGG_dir_list:
        tumor_type_dict[patientID] = 0 if patientID in HGG_dir_list else 1

    # === Train model ===
    train_model(partition, tumor_type_dict, survival_data,
                mode="single")      # 1 prediction
    train_model(partition, tumor_type_dict, survival_data,
                mode="tumortype")   # 2 predictions
    train_model(partition, tumor_type_dict, survival_data,
                mode="all")         # 3 predictions

    # === Evaluate model and export predictions ===
    predictions = evaluate_model(
        partition, tumor_type_dict, survival_data, mode="all")
    id_list = partition['holdout']
    # segmentation predictions (first output)
    export_tiff_predictions(predictions[0], id_list)

    # === Export ground truth for the same IDs ===
    export_ground_truth(id_list)
