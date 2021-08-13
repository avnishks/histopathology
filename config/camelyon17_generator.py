import importlib

import numpy as np
import tensorflow.keras as K


# Wrapper class to make everything fit in the dataGenerator framework.
# Take any basic module in parameters (mnist, cifar10, etc.) and convert it in the datagenerator.
class Camelyon17DataGenerator(data):
    def __init__(self, file_path):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
        """
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        self.summary()

    def __getitem__(self, index):
        with h5py.File(self.file_path,'r') as hdf5_file:
            img = hdf5_file['imgs'][index]
            coord = hdf5_file['coords'][index]
        
        img = Image.fromarray(img)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord

    def __len__(self):
        return self.length
