# src/evaluate.py
import os
import pickle
import numpy as np
from keras.models import load_model

from src.data import SingleDataGenerator, DataGenerator, SubDataGenerator
from src.model import UNET3D_model, weighted_dice_coefficient_loss, dice_coefficient, weighted_dice_coefficient


def evaluate_model(partition, tumor_type_dict, survival_data, mode="single"):
    params = {'dim': (160, 192, 160),
              'batch_size': 1,
              'n_classes': 3,
              'n_channels': 4,
              'shuffle': False}

    # Select generator and model structure
    if mode == "single":
        validation_generator = SingleDataGenerator(
            partition['holdout'], **params)
        model = isensee2017_model(input_shape=(4, 160, 192, 160), n_base_filters=12,
                                  depth=5, dropout_rate=0.3, n_segmentation_levels=3, n_labels=3)
        model.load_weights("./weights/model_1_weights.h5")
    elif mode == "tumortype":
        validation_generator = DataGenerator(
            partition['holdout'], tumor_type_dict, **params)
        model = isensee2017_model(input_shape=(4, 160, 192, 160), n_base_filters=12, depth=5,
                                  dropout_rate=0.3, n_segmentation_levels=3, n_labels=3, multihead="tumortype")
        model.load_weights("./weights/model_2_weights.h5")
    elif mode == "all":
        validation_generator = SubDataGenerator(
            partition['holdout'], tumor_type_dict, survival_data, **params)
        model = isensee2017_model(input_shape=(4, 160, 192, 160), n_base_filters=12,
                                  depth=5, dropout_rate=0.3, n_segmentation_levels=3, n_labels=3, multihead="all")
        model.load_weights("./weights/model_3_weights.h5")
    else:
        raise ValueError("Invalid mode")

    # Run prediction
    predictions = model.predict_generator(generator=validation_generator)
    # Save predictions
    with open(f"./weights/predictions_{mode}.pkl", "wb") as f:
        pickle.dump(predictions, f)
    print("Predictions saved to ./weights/")

    return predictions
