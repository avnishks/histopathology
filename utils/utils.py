import os
import sys
import cv2
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py

import PIL
from PIL import Image, ImageDraw, ImageFont

import openslide
from openslide import OpenSlideError, OpenSlideUnsupportedFormatError

import skimage.filters as sk_filters
from skimage import io, morphology, color


def open_slide(filename):
    """
    Create Openslide Object from a WSI (.svs, .tif, .ndpi, etc).
    
    Args:
        filename (str): Name of the slide file to open.
    
    Returns:
        An OpenSlide object representing a whole-slide image.
    """
    try:
        slide = openslide.open_slide(filename)
    except OpenSlideUnsupportedFormatError:
        slide = None
        print("Unrecognized file format.")
    except OpenSlideError:
        slide = None
        print("Slide file recognized. OpenSlide error.")
    except FileNotFoundError:
        slide = None
        print("Slide file not found.")
    return slide


def get_slide_info(filename):
    """
    Display information about the magnification levels in the WSI:
    Level count: Number of different image resolutions available within the WSI.
    Level dimensions: Pixel dimensions of the entire grid at a particular resolution level.
    
    Args:
        filename (str): Name of the slide file.

    Returns:
        A tuple of level_count and level_dimensions.
    """
    
    slide = openslide.open_slide(filename)
    level_count = slide.level_count
    level_dimensions = slide.level_dimensions
    # print("Level count:         " + str(level_count))
    # print("Level dimensions:    " + str(level_dimensions))
    return (level_count, level_dimensions)


def read_slide(slide, location, level, size, as_float=False):
    """
    Read a region of the WSI with Top Left corner at (x, y) at a certain resolution level
    
    Args:
        slide: Openslide Object
        location (tuple): (x, y) tuple giving the top left pixel in the level 0 reference frame
        level: resolution level
        size (tuple): (width, height) tuple giving the region size
    """
    img = slide.read_region(location, level, size)
    img = img.convert('RGB') # drop the alpha channel
    if as_float:
        img = np.asarray(img, dtype=np.float32)
    else:
        img = np.asarray(img)
    assert img.shape == (size[1], size[0], 3)
    return img


def display_slide(filename, level):
    """
    Display a scaled down version of a WSI.
    
    Args:
        filename (str): name of the WSI to display.
        level: resolution level
    """
    image_slide = open_slide(filename)
    image = read_slide(image_slide, 
                        (0,0), 
                        level, 
                        (image_slide.level_dimensions[level][0], image_slide.level_dimensions[level][1])
                        )

    plt.figure(figsize=(10, 10), dpi=100)
    plt.title("showing slide for: {}".format(os.path.splitext(os.path.basename(filename))[0]))
    plt.imshow(image)


def save_image(img_filename, img):
    """
    Save image to disk.

    Args:
        img_filename(str): name of the image to be saved.
        img = image data that is to eb saved.
    """
    # img = np.array(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(img_filename, img)
    cv2.imwrite(img_filename, img.astype(np.float32))


def pil_to_np_rgb(pil_img):
    """
    Convert a PIL Image to a NumPy array.
    Note that RGB PIL (w, h) -> NumPy (h, w, 3).
    
    Args:
        pil_img: The PIL Image.
    
    Returns:
        The PIL image converted to a NumPy array.
    """
    rgb = np.asarray(pil_img)
    return rgb


def filter_otsu_threshold(np_img, output_type="uint8"):
    """
    Compute Otsu threshold on image as a NumPy array and return binary image based on 
    pixels above threshold.
    
    Args:
        np_img: Image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).
    
    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel 
        above Otsu threshold.
    """
    otsu_thresh_value = sk_filters.threshold_otsu(np_img)
    otsu = (np_img > otsu_thresh_value)
    if output_type == "bool":
        pass
    elif output_type == "float":
        otsu = otsu.astype(float)
    else:
        otsu = otsu.astype("uint8") * 255
    return otsu


def overlay_img_and_segmentation_mask(img, mask):
    """
    Display the segmentation mask on top of the image.
    
    Args:
        img: Image to be segmented.
        mask: Segmented mask of the Image.
    """
    # Construct RGB version of gray-level mask
    mask_color = np.dstack((mask, mask, mask))

    # Convert the input image and color mask to the HSV colorspace
    img_hsv = color.rgb2hsv(img)
    mask_hsv = color.rgb2hsv(mask_color)

    # Replace the hue and saturation of the original image with that of the color mask
    mask_hsv[..., 0] = img_hsv[..., 0]
    mask_hsv[..., 1] = img_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(mask_hsv)

    # Display the output
    f, (ax0, ax1, ax2) = plt.subplots(1, 3,figsize=(60, 60), subplot_kw={'xticks':[], 'yticks':[]})
    ax0.imshow(img, cmap=plt.cm.gray)
    ax1.imshow(mask)
    ax2.imshow(img_masked)
    plt.show()


def display_segmentation_pipeline(slide, level):
    """
    Display the intermediate stages of the preprocessing pipeline.
    
    Args:
        slide: WSI to be segmented.
        level: Resolution level of the WSI.
    """
    
    image_slide = open_slide(slide)
    img = read_slide(image_slide,
                     (0,0), 
                     level, 
                     (image_slide.level_dimensions[level][0], image_slide.level_dimensions[level][1])
                     ).copy()
    img_clean = img.copy()
    img_clean[np.where((img==[0,0,0]).all(axis=2))] = [255,255,255] #remove back background
    
    img_hsv = cv2.cvtColor(img_clean, cv2.COLOR_RGB2HSV)
    img_med = cv2.medianBlur(img_hsv[:, :, 1], 7)

    otsu = filter_otsu_threshold(img_med)
    
    arr = otsu>0
    mask = morphology.remove_small_objects(arr, min_size=4)
    mask = morphology.remove_small_holes(mask, area_threshold=16)
    mask = morphology.opening(mask, morphology.disk(4))
    
    #overlay img and mask
    mask_color = np.dstack((mask, mask, mask))
    img_hsv = color.rgb2hsv(img_clean)
    mask_hsv = color.rgb2hsv(mask_color)
    # Replace the hue and saturation of the original image with that of the color mask
    mask_hsv[..., 0] = img_hsv[..., 0]
    mask_hsv[..., 1] = img_hsv[..., 1] * alpha
    img_masked = color.hsv2rgb(mask_hsv)
    
    #plot    
    fig, axs = plt.subplots(2, 3, figsize=(50, 50),)
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Input WSI')   
    axs[0, 1].imshow(img_clean)
    axs[0, 1].set_title('Removing Black Background') 
    axs[0, 2].imshow(img_hsv[:, :, 1])
    axs[0, 2].set_title('Saturation Component of HSV')
    axs[1, 0].imshow(img_med)
    axs[1, 0].set_title('Median Filtering')
    axs[1, 1].imshow(otsu)
    axs[1, 1].set_title('Otsu Thresholding') 
    axs[1, 2].imshow(mask)
    axs[1, 2].set_title('Post Morphological Operation')
    
    plt.show()

def read_labels(label_filepath):
    """ 
    Reads the slide-level labels from the Camelyon17's stage_labels.csv file and clean it.
    
    Args:
        
    Returns:
        labelDict : dictionary of {slide_name: tumor_label} for all the slides in training folder
    """
    
    myDict = {y[0]: y[1] for y in [x.split(",") for x in open(label_filepath).read().split('\n') if x]}
    labelDict = {}

    for key, val in myDict.items():
        if '.tif' in key:
            labelDict[key] = val

    newDict = labelDict
    for key in list(labelDict):
        newDict[key.split('.')[0]] = newDict.pop(key) #remove '.tif' from keys
    
    for key, value in newDict.items():
        if value == 'negative':
            newDict[key] = 0
        if value == 'itc':
            newDict[key] = 1
        if value == 'micro':
            newDict[key] = 2
        if value == 'macro':
            newDict[key] = 3
            
    return newDict


def save_to_hdf5(file_name, patches, coords, labels, db_location):
    """ 
    Saves the numpy arrays to HDF5 files. All patches from a single WSI will be saved 
    to the same HDF5 file, regardless of the transaction size specified by rows_per_txn, 
    because this is the most efficient way to use HDF5 datasets.
    
    Args:
        - slide        :    file path to the WSI to be saved into hdf5
        - db_location  :    folder to save h5 data in
        - patches      :    list of numpy images
        - coords       :    x, y tile coordinates
        - file_name    :    original source WSI name
        - labels       :    dictionary of {slide_name: tumor_label} for all the slides in training folder
    """
    print("Saving patches inHDF5 for {}.".format(os.path.basename(file_name)))
    
    if not os.path.exists(db_location):
        os.makedirs(db_location)
        
    # Save patches into hdf5 file.
    slide = file_name
    with h5py.File(db_location + 'training.h5','a') as hf:
        patient_index = "_".join(os.path.basename(slide).split('.')[0].split('_')[:2])
        slide_index = "_".join(os.path.basename(slide).split('.')[0].split('_')[3])
        slide_label = labels[os.path.basename(slide)]
#             grp = hf.create_group(patient_index)
        grp = hf.require_group(patient_index)
        subgrp = grp.require_group('wsi_{}'.format(slide_index))
#         subgrp.attrs["slide_label"] = slide_label
        
        for i, patch in enumerate(patches):
#             patch_name = file_name + "_" + str(i) 
            subsubgrp = subgrp.require_group('patch_{}'.format(i))
            subsubgrp.create_dataset('image', np.shape(patch), data=patch, compression="gzip", compression_opts=7)
            subsubgrp.create_dataset('label', np.shape(slide_label), data=slide_label)
            subsubgrp.attrs["patch_coords"] = (coords[i][0], coords[i][1])


#     Save all label meta into a csv file.
#     with open(db_location + file_name + '.csv', 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         for i in range(len(labels)):
#             writer.writerow([coords[i][0], coords[i][1], labels[i]])


def save_to_disk(patches, coords, file_name, labels, db_location):
    """ 
    Saves numpy patches to .png files (full resolution). 
    Meta data is saved in the file name.
    
    Args:
        - db_location  :    folder to save images in
        - patches      :    numpy images
        - coords       :    x, y tile coordinates
        - file_name    :    original source WSI name
        - labels       :    patch labels (opt)
    """
    print("Saving patches to disk (png).")
    
    if not os.path.exists(db_location):
        os.makedirs(db_location)
        
    save_labels = len(labels)
    for i, patch in enumerate(patches):
        patch_fname = file_name + "_" + str(coords[i][0]) + "_" + str(coords[i][1]) + "_"

        if save_labels:
            patch_fname += str(labels[i])
            
        Image.fromarray(patch).save(db_location + patch_fname + ".png")