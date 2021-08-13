import os
import sys
import cv2
import math
import glob
import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image, ImageDraw, ImageFont

import openslide
from openslide import OpenSlideError, OpenSlideUnsupportedFormatError

import skimage.filters as sk_filters
from skimage import io, morphology, color

from utils import open_slide, read_slide, pil_to_np_rgb, filter_otsu_threshold

def segment_tissue_from_background(slide, level):
	"""
	Segment the Tissue foreground from the background through a series of transformation:
	RGB --> Remove black background pixels--> HSV --> Median Blurring -->
	Thresholding --> Morphological Operations to fill holes

	NOTE: In some of the CAMELYON17 cases, the Otsu’s thresholding failed because of the
	black regions in the WSI. So before the application of image thresholding operation,
	the black pixel regions in the WSI background are replaced with white pixels.

	Args:
	    slide: Path to the slide to segment.

	Returns:
	    cleaned: a ndarray (n=2) containing a binary Tissue mask.
	"""
	image_slide = open_slide(slide)
	img = read_slide(image_slide, 
					(0,0), 
					level, 
					(image_slide.level_dimensions[level][0], image_slide.level_dimensions[level][1])
					).copy()
	#remove black background in some WSI
	img[np.where((img==[0,0,0]).all(axis=2))] = [255,255,255]

	img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	img_med = cv2.medianBlur(img_hsv[:, :, 1], 7)

	otsu = filter_otsu_threshold(img_med)
	arr = otsu>0
	# cleaned = morphology.dilation(arr)
	cleaned = morphology.remove_small_objects(arr, min_size=4)
	cleaned = morphology.remove_small_holes(cleaned, area_threshold=16)
	cleaned = morphology.opening(cleaned, morphology.disk(4))
	return cleaned