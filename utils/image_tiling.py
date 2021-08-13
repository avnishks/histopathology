import os
import sys
import glob
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
# import random

from openslide.deepzoom import DeepZoomGenerator
from ai.histopath_template.utils.notebook_refactor.utils import open_slide, save_to_hdf5, save_to_disk, read_labels


# PATCH_DIR_H5 = os.path.join(".", "datasets/Camelyon17/patches/h5/")
# PATCH_DIR_PNG = os.path.join(".", "datasets/Camelyon17/patches/png/")
# STITCH_DIR = os.path.join(".", "datasets/Camelyon17/stitches/")
# PATCH_LEVEL = 12
# PATCH_SIZE = 256
# PIXEL_OVERLAP = 64 #overlap in patch extraction
# STORAGE_OPTION = 'hdf5' #choose either one of 'hdf5' or 'disk'

# seed = random.randrange(sys.maxsize)
# random.seed(seed)
# print(f"random seed: {seed}")


def patch_to_tile_size(patch_size, overlap):
    return patch_size - overlap*2


def sample_and_store_patches(slide_path,
                            label_filepath,
                            level,
                            patch_dir_h5="",
                            patch_dir_png="",
                            pixel_overlap=0,
                            patch_size=256,
                            limit_bounds=False,
                            rows_per_txn=200,
                            storage_option='hdf5'):
    """ 
    Sample patches of specified size from WSI file.
    
    Args:
        - slides          :    list of all available WSI file paths
        - label_filepath  :    path to file containing WSI level labels (csv)
        - file_name       :    name of whole slide image to sample from
        - file_dir        :    directory file is located in
        - pixel_overlap   :    pixels overlap on each side
        - level           :    0 is lowest resolution; level_count - 1 is highest
        - rows_per_txn    :    how many patches to load into memory at once
        - storage_option  :    the patch storage option              
        
    Note: patch_size is the dimension of the sampled patches, NOT equivalent to 
    openslide's definition of tile_size.
    
    Return:
        Count(int): #tiles extracted from the WSI
    """
    tile_size = patch_to_tile_size(patch_size, pixel_overlap)
    labels = read_labels(label_filepath)

    for file in slide_path:
        slide = open_slide(file)  
        tiles = DeepZoomGenerator(slide,    
                                tile_size=tile_size,
                                overlap=pixel_overlap,
                                limit_bounds=limit_bounds)

        if level >= tiles.level_count:
            print("[py-wsi error]: requested level does not exist. Number of slide levels: " 
                  + str(tiles.level_count))
            return 0

        x_tiles, y_tiles = tiles.level_tiles[level]
        # print("x_tiles, y_tiles: ", x_tiles, y_tiles)
        # print("level count: ", slide.level_count)

        x, y = 0, 0
        count, batch_count = 0, 0
        patches, coords = [], []
        while y < y_tiles:
            while x < x_tiles:
                new_tile = np.array(tiles.get_tile(level, (x, y)), dtype=np.uint8)
                # OpenSlide calculates overlap in such a way that sometimes depending on the 
                # dimensions, edge patches are smaller than the others. We will ignore such patches.
                if np.shape(new_tile) == (patch_size, patch_size, 3):
                    patches.append(new_tile)
                    coords.append(np.array([x, y]))
                    count += 1

                x += 1

            # To save memory, we will save data into the dbs every rows_per_txn rows. i.e., 
            # each transaction will commit #rows_per_txn rows of patches. Write after last row 
            # regardless. HDF5 does NOT follow this convention due to efficiency.
            if (y % rows_per_txn == 0 and y != 0) or y == y_tiles-1:
                if storage_option == 'disk':
                    file_name = os.path.basename(slide_path).rsplit('.')[0]
                    save_to_disk(patches, coords, file_name, labels, db_location=patch_dir_png)
                if storage_option != 'hdf5':
                    del patches
                    del coords
                    patches, coords = [], [] # Reset right away.

            y += 1
            x = 0

        if storage_option == 'hdf5':
            file_name = os.path.basename(file).split('.')[0]
            save_to_hdf5(file_name, patches, coords, labels, db_location=patch_dir_h5)

    return count