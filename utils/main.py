import os
import sys
import cv2
import math
import glob
import click
import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image, ImageDraw, ImageFont

import openslide
from openslide import OpenSlideError, OpenSlideUnsupportedFormatError

import skimage.filters as sk_filters
from skimage import io, morphology, color

# from utils import open_image, pil_to_np_rgb, filter_otsu_threshold, display_img
from utils import get_slide_info, display_slide, save_image, read_labels
from image_segmentation import segment_tissue_from_background
from image_tiling import sample_and_store_patches


## TO-DO: convert to Click
# BASE_DIR = os.path.join("/opt/imagia/data/", "datasets/public/Camelyon17/")
# MASK_DIR = os.path.join(BASE_DIR, "masks/")
# LEVEL = 8 # 0-9 (or 0-8) in (some) Chamelyon17 tif images; ideally set to level_count-1 for each slide

# PATCH_DIR_H5 = os.path.join(".", "datasets/Camelyon17/patches/h5/")
# PATCH_DIR_PNG = os.path.join(".", "datasets/Camelyon17/patches/png/")
# STITCH_DIR = os.path.join(".", "datasets/Camelyon17/stitches/")
# PATCH_LEVEL = 12
# PATCH_SIZE = 256
# PIXEL_OVERLAP = 64 #overlap in patch extraction
# STORAGE_OPTION = 'hdf5' #choose either one of 'hdf5' or 'disk'

# label_filepath = "./datasets/Camelyon17/training/stage_labels.csv"


if __name__ == '__main__':
	slide_paths = glob.glob(base_dir + 'training/center_*/*.tif')
	# print("Slides: \n", slides)
	if not os.path.exists(mask_dir): os.mkdir(mask_dir)

	for slide in slide_paths:
		level_count, level_dimensions = get_slide_info(slide)
		print("Level count for {}: ".format(os.path.splitext(os.path.basename(slide))[0]), level_count)
		# print("Level Dimensions count for {}: ".format(os.path.splitext(os.path.basename(slide))[0]), level_dimensions)
		# display_slide(slide, LEVEL)

		mask = segment_tissue_from_background(slide, LEVEL)
		# print("mask type: ", type(mask))
		mask_filename = os.path.join(mask_dir, os.path.splitext(os.path.basename(slide))[0])
		save_image('{}.png'.format(mask_filename), mask)

	#patching and save as HDF5
	# labels = read_labels(label_filepath)
	sample_and_store_patches(slide_paths,
							label_filepath,
							level=PATCH_LEVEL,
							pixel_overlap=PIXEL_OVERLAP,
	                        patch_size=PATCH_SIZE,
	                        limit_bounds=True,
	                        rows_per_txn=200,
	                        storage_option=STORAGE_OPTION)