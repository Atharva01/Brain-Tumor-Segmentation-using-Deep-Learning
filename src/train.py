# src/train.py
import keras
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import pandas as pd

from src.data import SingleDataGenerator, DataGenerator, SubDataGenerator
from src.model import UNET3D_model, weighted_dice_coefficient_loss, dice_coefficient, weighted_dice_coefficient


def train_model(partition, tumor_type_dict, survival_data, mode="single"):
    # Parameters
    params = {'dim': (160, 192, 160),
              'batch_size': 1,
              'n_classes': 3,
              'n_channels': 4,
              'shuffle': True}

    # Choose data generator and model outputs
    if mode == "single":
        training_generator = SingleDataGenerator(partition['train'], **params)
        validation_generator = SingleDataGenerator(partition['test'], **params)
        model = UNET3D_model(input_shape=(4, 160, 192, 160), n_base_filters=12,
                             depth=5, dropout_rate=0.3, n_segmentation_levels=3, n_labels=3)
        model.compile(optimizer=RMSprop(lr=5e-4),
                      loss={'activation_block': weighted_dice_coefficient_loss},
                      loss_weights={'activation_block': 1.},
                      metrics={'activation_block': ['accuracy', weighted_dice_coefficient, dice_coefficient]})
        weights_file = "./weights/1pred_weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        model_file = "./weights/model_1_weights.h5"
        history_file = "./weights/history_1_pred.pkl"
    elif mode == "tumortype":
        training_generator = DataGenerator(
            partition['train'], tumor_type_dict, **params)
        validation_generator = DataGenerator(
            partition['test'], tumor_type_dict, **params)
        model = UNET3D_model(input_shape=(4, 160, 192, 160), n_base_filters=12, depth=5,
                             dropout_rate=0.3, n_segmentation_levels=3, n_labels=3, multihead="tumortype")
        model.compile(optimizer=RMSprop(lr=5e-4),
                      loss={'activation_block': weighted_dice_coefficient_loss,
                            'tumortype_block': 'binary_crossentropy'},
                      loss_weights={'activation_block': 1.,
                                    'tumortype_block': 0.2},
                      metrics={'activation_block': ['accuracy', weighted_dice_coefficient, dice_coefficient], 'tumortype_block': ['accuracy']})
        weights_file = "./weights/2pred_weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        model_file = "./weights/model_2_weights.h5"
        history_file = "./weights/history_2_pred.pkl"
    elif mode == "all":
        training_generator = SubDataGenerator(
            partition['train'], tumor_type_dict, survival_data, **params)
        validation_generator = SubDataGenerator(
            partition['test'], tumor_type_dict, survival_data, **params)
        model = UNET3D_model(input_shape=(4, 160, 192, 160), n_base_filters=12,
                             depth=5, dropout_rate=0.3, n_segmentation_levels=3, n_labels=3, multihead="all")
        model.compile(optimizer=RMSprop(lr=5e-4),
                      loss={'activation_block': weighted_dice_coefficient_loss,
                            'survival_block': 'mean_squared_error', 'tumortype_block': 'binary_crossentropy'},
                      loss_weights={'activation_block': 1.,
                                    'survival_block': 0.2, 'tumortype_block': 0.2},
                      metrics={'activation_block': ['accuracy', weighted_dice_coefficient, dice_coefficient],
                               'survival_block': ['accuracy', 'mae'],
                               'tumortype_block': ['accuracy']})
        weights_file = "./weights/3pred_weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        model_file = "./weights/model_3_weights.h5"
        history_file = "./weights/history_3_pred.pkl"
    else:
        raise ValueError("Invalid mode")

    cb_1 = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    cb_2 = ModelCheckpoint(filepath=weights_file, monitor='val_loss',
                           save_best_only=True, save_weights_only=False, mode='auto')
    results = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=100,
                                  workers=4,
                                  callbacks=[cb_1, cb_2])
    model.save_weights(model_file)
    with open(history_file, "wb") as f:
        pickle.dump(results.history, f)
    print("Saved model and history to disk")
