import argparse
import gc
import json
import logging
import sys
import time
import os
import numpy as np
import torch
from itertools import chain

import logging
import os
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
import monai
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.data import ImageDataset, DataLoader,decollate_batch,CSVSaver
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd
from generate_transforms import*
from utils import*
from classification_models import*



'''
python test.py \
-c /data/usr/ft42/CVIT_XAI/LungRADS_Modeling/LungRADS_FPR/BaselineCNNs/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/config_train.json \
-m /data/usr/ft42/CVIT_XAI/LungRADS_Modeling/LungRADS_FPR/BaselineCNNs/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/saved_models/ \
-csv /data/usr/ft42/CVIT_XAI/LungRADS_Modeling/numpy_Datasets/LungRADs_Resampled_numpy_CV4/Fold1/result_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_withGT_fold1_numpy.csv \
-dc "unique_Annotation_id_nifti" \
-dp /data/usr/ft42/CVIT_XAI/LungRADS_Modeling/numpy_Datasets/LungRADs_Resampled_numpy_CV4/Fold1/Fold1_patch64x64y64z/nifti/ \
-op /data/usr/ft42/CVIT_XAI/LungRADS_Modeling/LungRADS_FPR/BaselineCNNs/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/result/ \
-o "resnet18_validation_output.csv" \
-b 2
'''



def test_main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("-c",      "--config-file",       help="config json file that stores hyper-parameters")
    parser.add_argument("-m",      "--model-weight",      help="Path to the model weight")
    parser.add_argument("-csv",    "--data-csv",          help="Path csv having the datalist")
    parser.add_argument("-dc",     "--data-column",       help="Path csv having the datalist")
    parser.add_argument("-dp",     "--data-path",         help="Path csv having the datalist")
    parser.add_argument("-op",      "--output-path",        help="Path to the output csv")
    parser.add_argument("-o",      "--output-name",         help="output csv name")
    parser.add_argument("-b",      "--testbatch-size",  type=int,  help="batch size")
    args = parser.parse_args()



    amp = True
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    config_dict = json.load(open(args.config_file, "r"))
    for k, v in config_dict.items():
        setattr(args, k, v)

    #---- Creating folders---#
    #---- Creating folders---#

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    #-| Testing
    test_df               = pd.read_csv(args.data_csv)
    test_df['data_path']  =  args.data_path+ test_df[args.data_column]
    test_images           = test_df['data_path'].tolist()
    test_files            = [{"img": img, "file_name":img} for img in test_images]



    # Define transforms for image
    test_transforms = generate_classification_test_transform(image_key      = config_dict["image_key"],img_patch_size  = config_dict["img_patch_size"])
    test_ds         = monai.data.Dataset(data=test_files,  transform=test_transforms)
    test_loader     = DataLoader(test_ds, batch_size=args.testbatch_size, num_workers=4, pin_memory=torch.cuda.is_available())
    # Create networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_classification_model(Model_name=config_dict["Model_name"],spatial_dims=config_dict["spatial_dims"],n_input_channels=config_dict["n_input_channels"], num_classes=config_dict["num_classes"],device=device)

    test_epoch_len   = len(test_ds) // test_loader.batch_size
    test_step_i      = 0


    prediction_List=[]
    id_list        =[]
    model.load_state_dict(torch.load(args.model_weight))
    model.eval()
    with torch.no_grad():
        #saver = CSVSaver(output_dir=args.output_path,filename=args.output_name)
        for test_data in test_loader:
            test_step_i += 1
            print(f"Predicting:{test_step_i}/{test_epoch_len}")
            test_images       = test_data["img"].to(device)
            result_modelOut   = model(test_images)
            softmax           = torch.nn.Softmax(dim=1)
            test_outputs2     = softmax(result_modelOut)
            prediction_List.append(test_outputs2[:,1].cpu().numpy())
            id_list.append(test_data["file_name"])
    torch.cuda.empty_cache()
    flattened_prediction = list(chain.from_iterable(prediction_List))
    flattened_filname    = list(chain.from_iterable(id_list))
    df_predict = pd.DataFrame(list(zip(flattened_filname,flattened_prediction)))
    prediction_save_path = args.output_path + args.output_name
    df_predict.to_csv(prediction_save_path,index=False, header=False,encoding='utf-8')

if __name__ == "__main__":
    test_main()
