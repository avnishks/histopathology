import numpy as np
import h5py
import tqdm
import random

from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input


# Wrapper class to make everything fit in the dataGenerator framework.
# Take any basic module in parameters (mnist, cifar10, etc.) and convert it in the datagenerator.
class Camelyon17DataGenerator(Sequence):
    """
        Camelyon17 data generator. Patches have been extracted and stored in an h5 file. We load it here and
        extract a batch from all the available patches from all patients.

        :param h5_file_path: path to h5 patch file
        :type h5_file_path: str
        :param batch_size: batch size
        :type batch_size: int
        :param n_classes: number of classes
        :type n_classes: int
        :param n_examples: debugging parameter, size of the dataset that should be used in number of examples
        :type n_examples: int
        :param preprocessing: the preprocessing to be used on the dataset, either None or 'imagenet'
        :type preprocessing: str
        :param augmentation: the augmentation to be used on the dataset
        :type augmentation: dict
        :param shuffle: True if data should be shuffled after each epoch
        :type shuffle: bool
        :param seed: seed to reproduce patches generation
        :type seed: int
    """
    def __init__(self, h5_file_path, batch_size=32, n_classes=2, n_examples=None, preprocessing=None, augmentation=None, shuffle=True, seed=1234):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
        """
        self.h5_file_path = h5_file_path
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_examples = n_examples
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.seed = seed

        # Set up seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        with h5py.File(self.h5_file_path, 'r') as h5:
            self.ids = [patient_id + '/' + patch_id for patient_id in h5 for patch_id in h5[patient_id]]
            #dset = h5['imgs']
            #self.length = len(dset)

        #self.summary()

        if self.n_examples is not None:
            self.ids = self.ids[:self.n_examples]

        if self.shuffle:
            np.random.shuffle(self.ids)
            
        if self.augmentation is not None:
            self.augmentation = ImageDataGenerator(**self.augmentation)


    def __getitem__(self, index):
        """
        Generates one batch of data

        :param index:
        :type index:
        :return: One batch of data
        :rtype: tuple
        """

        _from = index * self.batch_size
        local_batch = self.ids[_from:_from + self.batch_size] if _from + self.batch_size < len(self.ids) else self.ids[_from:]

        X, y = [], []
        with h5py.File(self.h5_file_path, 'r') as h5:
            for local_id in local_batch:
                X.append(np.expand_dims(h5[local_id + '/img'][()], 0))
                y.append(h5[local_id + '/label'][()])
                # coord = h5[local_id].attrs['coords']
                # While waiting for the complete h5, still using the single patient one:
                # X.append(np.expand_dims(h5[local_id][()], 0).astype(np.float32))
                # y.append(np.random.randint(2))
        
        X = np.concatenate(X)
        y = np.array(y)
        #img = Image.fromarray(img)

        # Apply preprocessing and augmentations to the whole batch
        # if self.target_patch_size is not None:
        #     img = img.resize(self.target_patch_size)
        # img = self.roi_transforms(img).unsqueeze(0)
        if self.preprocessing=='imagenet':
            X = preprocess_input(X, data_format='channels_last', mode='tf')
        else:
            X = X / 255.0

        if self.augmentation is not None:
            #X = self.augmentation(X)
            for idx in range(len(X)):
                X[idx] = self.augmentation.random_transform(X[idx])

        # reshuffle for next epoch
        if self.shuffle and _from + self.batch_size > len(self.ids):
            np.random.shuffle(self.ids)
        

        return X, to_categorical(y, self.n_classes)


    def __len__(self):
        """
        :return: Number of batches per epochs
        :rtype: int
        """
        return int(np.floor(len(self.ids) / self.batch_size)) # ceil or floor? ceil -> final batch incomplete | floor -> neglect final incomplete batch
