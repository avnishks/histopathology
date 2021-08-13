import os
import sys
import cv2
import math
import glob
import click
import numpy as np
import matplotlib.pyplot as plt
import yaml

import PIL
from PIL import Image, ImageDraw, ImageFont

import openslide
from openslide import OpenSlideError, OpenSlideUnsupportedFormatError

import skimage.filters as sk_filters
from skimage import io, morphology, color

import ai.histopath_template.training as training
import ai.histopath_template.utils.configuration as configuration
import click
from oi_core.build import SUPPORTED_CONFIG_TYPES
from oi_core import add_config_tree, add_definitions_tree

from . import definitions

import ai.histopath_template.utils.notebook_refactor.utils as utils
# import ai.histopath_template.utils.notebook_refactor.image_segmentation as segmenting
import ai.histopath_template.utils.notebook_refactor.image_tiling as tiling

#####
# from utils import get_slide_info, display_slide, save_image, read_labels
# from image_segmentation import segment_tissue_from_background
# from image_tiling import sample_and_store_patches

#####

# add_definitions_tree(definitions)  # connecting the project to oi-core

# Enable click
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'


# def load_config(path: str) -> dict:
#     """Load the config file.

#     Load config. Supported config extensions are provided as keys in 
#     `SUPPORTED_CONFIG_TYPES`.

#     :param path: path to the config
#     :type path: str
#     :return: the config, as a dict
#     :rtype: dict
#     """
#     extension = path.split(".")[-1]
#     if extension in SUPPORTED_CONFIG_TYPES:
#         return SUPPORTED_CONFIG_TYPES[extension](path)
#     else:
#         raise TypeError(f"Config extensions: {extension} not supported")

def load_config(config_file):

    default_config = {'cuda': True,
                      'seed': 1234
    }

    with open(config_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.Loader)

    return {**default_config, **yaml_cfg}


@click.group()
def run():
    pass


@run.command()
@click.option('--config', '-cgf', type=click.Path(exists=True, resolve_path=True), help='Configuration file.')
def train(config):
    cfg = load_config(config)
    training.train(cfg)


@run.command()
@click.option('--config', '-cgf', type=click.Path(exists=True, resolve_path=True), help='Configuration file.')
@click.option('--n-iter', type=int, default=10, help='Configuration file.')
@click.option('--base-estimator', type=click.Choice(["GP", "RF", "ET", "GBRT"]), default="GP", help='Estimator.')
@click.option('--n-initial-points', type=int, default=10, help='Number of evaluations of func with initialization points before approximating it with base_estimator.')
@click.option('--random-state', type=int, default=1234, help='Set random state to something other than None for reproducible results.')
@click.option('--train-function', type=str, default="train", help='Training function to optimize over.')
def train_skopt(config, n_iter, base_estimator, n_initial_points, random_state, train_function):
    cfg = load_config(config)
    training.train_skopt(cfg, n_iter=n_iter,
                    base_estimator=base_estimator,
                    n_initial_points=n_initial_points,
                    random_state=random_state,
                    train_function=getattr(training, train_function))

@run.command()
@click.option('--slide_paths', type=click.Path(exists=True, resolve_path=True), help='Configuration file.')
@click.option('--level', type=int, default=8, help='Configuration file.')
@click.option('--pixel_overlap', type=int, default=64, help='Configuration file.')
@click.option('--patch_size', type=int, default=256, help='Configuration file.')
@click.option('--limit_bounds', type=click.Choice(["True", "False"]), default="False", help='Estimator.')
@click.option('--rows_per_txn', type=int, default=200, help='Configuration file.')
@click.option('--storage_options', type=click.Choice(["hdf5", "disk"]), default="hdf5", help='Estimator.')
def extract_patch_manual(slide_path, level, pixel_overlap, patch_size, limit_bounds, rows_per_txn, storage_option):
    tiling.sample_and_store_patches(slide_paths,
                                    level=level,
                                    pixel_overlap=pixel_overlap,
                                    patch_size=patch_size,
                                    limit_bounds=limit_bounds,
                                    rows_per_txn=rows_per_txn,
                                    storage_option=storage_option)

@run.command()
@click.option('--config', '-cgf', type=click.Path(exists=True, resolve_path=True), help='Configuration file.')
def extract_patch(config):
    cfg = load_config(config)
    slide_paths = glob.glob(cfg['base_dir'] + 'training/center_*/*.tif')
    tiling.sample_and_store_patches(slide_paths, **cfg['tiling_arguments'])


def main():
    run()

if __name__ == '__main__':
    main()
