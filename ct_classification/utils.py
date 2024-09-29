import numpy as np
import pandas as pd
import json
import logging
import os
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import ImageDataset, DataLoader,decollate_batch,CSVSaver
from monai.metrics import compute_roc_auc
from monai.metrics import ROCAUCMetric
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

import random
random.seed(200)


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"The folder has been created at: {path}")
    else:
        print(f"The folder already exists at: {path}")


# Assuming all imports are done as in your original code
class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def compute_class_weights(labels):
    """
    Compute class weights inversely proportional to the frequency of each class.

    Parameters:
        labels (numpy.array): Array of labels.

    Returns:
        numpy.array: Weights for each class.
    """
    # Count the frequency of each class
    class_counts = np.bincount(labels)
    # Compute weights inversely proportional to these frequencies
    weights = 1. / class_counts
    # Normalize weights so that they sum up to 1
    normalized_weights = weights / weights.sum()
    return normalized_weights

def filter_and_sample_rows(df, class_column, sampling_times, random_state=None):
    """
    Filter and sample rows from the DataFrame based on the class_column.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.
        class_column (str): Name of the column containing class information.
        random_state (int or None, optional): Random state to replicate results. Default is None.

    Returns:
        pandas.DataFrame: A new DataFrame containing all positive rows and 2 times randomly selected negative rows.
    """
    # Step 1: Separate positive and negative rows
    positive_rows = df[df[class_column] == 1]
    negative_rows = df[df[class_column] == 0]
    # Step 2: Keep all positive rows
    result_df = positive_rows.copy()
    # Step 3: Randomly select 2 times the number of positive rows and keep them
    num_negative_rows_to_keep = sampling_times * len(positive_rows)
    selected_negative_rows = negative_rows.sample(n=num_negative_rows_to_keep, random_state=random_state)
    # Concatenate the positive rows with the selected negative rows
    result_df = pd.concat([result_df, selected_negative_rows])
    # Shuffle the resulting DataFrame to randomize the order
    result_df = result_df.sample(frac=1, random_state=random_state)
    result_df = result_df.reset_index(drop=True)
    return result_df