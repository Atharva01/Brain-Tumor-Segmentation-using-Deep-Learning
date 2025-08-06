# src/model.py
import numpy as np
from functools import partial
from keras import backend as K
from keras.layers import (Input, Conv3D, MaxPooling3D, UpSampling3D, Activation,
                          BatchNormalization, LeakyReLU, Add, SpatialDropout3D,
                          GlobalAveragePooling3D, Dense)
from keras.engine import Model
from keras.optimizers import RMSprop

# Dice Coefficient and Losses


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=1e-5):
    return K.mean(2. * (K.sum(y_true * y_pred, axis=axis) + smooth/2) /
                  (K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)

# 3D UNet building blocks


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=LeakyReLU, padding='same', strides=(1, 1, 1)):
    layer = Conv3D(n_filters, kernel, padding=padding,
                   strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    if activation is not None:
        return activation()(layer)
    else:
        return layer


def create_localization_module(input_layer, n_filters):
    conv1 = create_convolution_block(input_layer, n_filters)
    conv2 = create_convolution_block(conv1, n_filters, kernel=(1, 1, 1))
    return conv2


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    conv1 = create_convolution_block(input_layer, n_level_filters)
    dropout = SpatialDropout3D(
        rate=dropout_rate, data_format=data_format)(conv1)
    conv2 = create_convolution_block(dropout, n_level_filters)
    return conv2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    conv = create_convolution_block(up_sample, n_filters)
    return conv

# Main 3D UNet model


def UNET3D_model(input_shape=(4, 160, 192, 160), n_base_filters=12, depth=5, dropout_rate=0.3, n_segmentation_levels=3, n_labels=3, activation_name="sigmoid", multihead=None):
    inputs = Input(input_shape)
    current_layer = inputs
    level_output_layers = []
    level_filters = []
    # Contracting path
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)
        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(
                current_layer, n_level_filters, strides=(2, 2, 2))
        context_output_layer = create_context_module(
            in_conv, n_level_filters, dropout_rate=dropout_rate)
        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    # Expansive path
    segmentation_layers = []
    for level_number in range(depth-2, -1, -1):
        up_sampling = create_up_sampling_module(
            current_layer, level_filters[level_number])
        concatenation_layer = K.concatenate(
            [level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(
            concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, create_convolution_block(
                current_layer, n_filters=n_labels, kernel=(1, 1, 1)))
    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])
        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)
    activation_block = Activation(
        activation_name, name='activation_block')(output_layer)

    # Multihead for tumor type/survival if needed
    outputs = [activation_block]
    if multihead == "tumortype" or multihead == "all":
        tumortype_conv_1 = Conv3D(filters=n_level_filters, kernel_size=(
            3, 3, 3), padding='same', strides=(1, 1, 1), name='tumortype_conv_1')(summation_layer)
        tumortype_conv_2 = Conv3D(filters=n_level_filters, kernel_size=(
            3, 3, 3), padding='same', strides=(1, 1, 1), name='tumortype_conv_2')(tumortype_conv_1)
        tumortype_dropout = SpatialDropout3D(
            rate=dropout_rate, data_format='channels_first', name='tumortype_dropout')(tumortype_conv_2)
        tumortype_conv_3 = Conv3D(filters=n_level_filters, kernel_size=(
            3, 3, 3), padding='same', strides=(1, 1, 1), name='tumortype_conv_3')(tumortype_dropout)
        tumortype_GAP = GlobalAveragePooling3D(
            name='tumortype_GAP')(tumortype_conv_3)
        tumortype_block = Dense(1, activation='sigmoid',
                                name='tumortype_block')(tumortype_GAP)
        outputs.append(tumortype_block)
    if multihead == "survival" or multihead == "all":
        survival_conv_1 = Conv3D(filters=n_level_filters, kernel_size=(
            3, 3, 3), padding='same', strides=(1, 1, 1), name='survival_conv_1')(summation_layer)
        survival_conv_2 = Conv3D(filters=n_level_filters, kernel_size=(
            3, 3, 3), padding='same', strides=(1, 1, 1), name='survival_conv_2')(survival_conv_1)
        dropout = SpatialDropout3D(
            rate=dropout_rate, data_format='channels_first', name='dropout')(survival_conv_2)
        survival_conv_3 = Conv3D(filters=n_level_filters, kernel_size=(
            3, 3, 3), padding='same', strides=(1, 1, 1), name='survival_conv_3')(dropout)
        survival_GAP = GlobalAveragePooling3D(
            name='survival_GAP')(survival_conv_3)
        survival_block = Dense(1, activation='linear',
                               name='survival_block')(survival_GAP)
        outputs.append(survival_block)

    model = Model(inputs=inputs, outputs=outputs)
    return model
