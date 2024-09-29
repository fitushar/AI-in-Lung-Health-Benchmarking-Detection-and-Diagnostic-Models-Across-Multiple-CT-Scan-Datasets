import torch
import numpy as np
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
    RandFlipd,
    RandSpatialCropd,
)

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
    Resized,
    ScaleIntensityRanged,
    EnsureTyped,
)


def generate_classification_train_transform(image_key,label_key,img_patch_size):

    '''
    image_key: "image"
    img_patch_size : patch size as numpy (64,64,64) [z,y,x]

    '''
    train_transforms = Compose(
    [
        LoadImaged(image_key),
        EnsureChannelFirstd(image_key,label_key,channel_dim="no_channel"),
        RandFlipd(image_key, spatial_axis=0, prob=0.5),
        RandFlipd(image_key, spatial_axis=1, prob=0.5),
        RandFlipd(image_key, spatial_axis=2, prob=0.5),
        RandSpatialCropd(image_key, roi_size=(img_patch_size[0]-8,img_patch_size[1],img_patch_size[2]-8)),
        Resized(image_key, spatial_size=img_patch_size, mode="trilinear", align_corners=True),
        RandRotate90d(image_key, prob=0.5, spatial_axes=[0, 1]),
        RandRotate90d(image_key, prob=0.5, spatial_axes=[0, 2]),
        EnsureTyped((image_key, label_key)),

    ])
    return train_transforms


def generate_classification_val_transform(image_key,label_key,img_patch_size):

    '''
    image_key: "image"
    img_patch_size : patch size as numpy (64,64,64) [z,y,x]
    '''
    val_transforms = Compose(
    [
        LoadImaged(image_key),
        EnsureChannelFirstd(image_key,channel_dim="no_channel"),
        Resized(image_key, spatial_size=img_patch_size, mode="trilinear", align_corners=True),
        EnsureTyped((image_key, label_key)),
    ])
    return val_transforms


def generate_classification_test_transform(image_key,img_patch_size):

    '''
    image_key: "image"
    img_patch_size : patch size as numpy (64,64,64) [z,y,x]

    '''
    test_transforms = Compose(
    [
        LoadImaged(image_key),
        EnsureChannelFirstd(image_key,channel_dim="no_channel"),
        Resized(image_key, spatial_size=img_patch_size, mode="trilinear", align_corners=True),
        EnsureTyped(image_key),
    ])
    return test_transforms
