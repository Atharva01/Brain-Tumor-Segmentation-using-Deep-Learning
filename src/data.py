# src/data.py
import os
import numpy as np
import pickle
import nibabel as nib
import keras

# Data Generator for segmentation (single prediction)


class SingleDataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=1, dim=(160, 192, 160), n_channels=4, n_classes=3, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size, 3, *self.dim))
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = pickle.load(open(f"./data/{ID}_images.pkl", "rb"))
            y[i,] = pickle.load(open(f"./data/{ID}_seg_mask_3ch.pkl", "rb"))
        return X, y

# Data Generator for segmentation + tumor type (two predictions)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, tumor_type_dict, batch_size=1, dim=(160, 192, 160), n_channels=4, n_classes=3, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.tumor_type_dict = tumor_type_dict
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y1, y2 = self.__data_generation(list_IDs_temp)
        return X, [y1, y2]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y1 = np.empty((self.batch_size, 3, *self.dim))
        y2 = np.empty(self.batch_size)
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = pickle.load(open(f"./data/{ID}_images.pkl", "rb"))
            y1[i,] = pickle.load(open(f"./data/{ID}_seg_mask_3ch.pkl", "rb"))
            y2[i,] = self.tumor_type_dict[ID]
        return X, y1, y2

# Data Generator for segmentation + tumor type + survival (three predictions)


class SubDataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, tumor_type_dict, survival_data, batch_size=1, dim=(160, 192, 160), n_channels=4, n_classes=3, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.tumor_type_dict = tumor_type_dict
        self.survival_data = survival_data
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y1, y2, y3 = self.__data_generation(list_IDs_temp)
        return X, [y1, y2, y3]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y1 = np.empty((self.batch_size, 3, *self.dim))
        y2 = np.empty(self.batch_size)
        y3 = np.empty(self.batch_size)
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = pickle.load(open(f"./data/{ID}_images.pkl", "rb"))
            y1[i,] = pickle.load(open(f"./data/{ID}_seg_mask_3ch.pkl", "rb"))
            y2[i,] = self.tumor_type_dict[ID]
            y3[i,] = self.survival_data[self.survival_data.Brats17ID ==
                                        ID].Survival.astype(int).values.item(0)
        return X, y1, y2, y3
