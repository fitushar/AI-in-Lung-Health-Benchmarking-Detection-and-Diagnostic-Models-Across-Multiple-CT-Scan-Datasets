import argparse
import gc
import json
import logging
import sys
import time
import os
import numpy as np
import torch

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
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd
from generate_transforms import*
from utils import*
from classification_models import*
from warmup_scheduler import GradualWarmupScheduler


def tain_main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("-c", "--config-file",help="config json file that stores hyper-parameters")
    parser.add_argument("-r", "--resume-checkpoint", default=None, help="Path to the checkpoint file from which to resume training")
    args = parser.parse_args()

    # Assuming rest of the initialization is done as in your original code
    set_determinism(seed=0)
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



    run_scr_save_path = config_dict["Model_save_path_and_utils"]
    model_path        = run_scr_save_path +'/saved_models/' + config_dict["run_prefix"]+str(config_dict["which_fold"])+'_'+ config_dict["Model_name"]+'.pt'
    tfevent_dir       = run_scr_save_path +'/tfevent_train/'+ config_dict["run_prefix"]+str(config_dict["which_fold"])+'_'+ config_dict["Model_name"]

    create_folder_if_not_exists(config_dict["Model_save_path_and_utils"]+'/result/')
    create_folder_if_not_exists(config_dict["Model_save_path_and_utils"]+'/tfevent_train/')
    create_folder_if_not_exists(config_dict["Model_save_path_and_utils"]+'/saved_models/')
    create_folder_if_not_exists(tfevent_dir)



    #####---Logfile
    log_file_path = model_path.split('.pt')[0]+'.log'
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()]
    )

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    # Redirect stdout and stderr to the log file
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)
    monai.config.print_config()



    # dataset creation
    #- Reading Training and Validation dataframes and sampling

    #-| Training
    train_df                    = pd.read_csv(config_dict["training_csv_path"])
    train_df['data_path']       = config_dict["training_nifti_dir" ] + train_df[config_dict["data_column_name"]]
    if config_dict["use_sampling"]:
        train_df      = filter_and_sample_rows(train_df, class_column=config_dict["label_column_name"], sampling_times=config_dict["sampling_ratio"], random_state=42)

    #-| Validation
    validation_df               = pd.read_csv(config_dict["validation_csv_path"])
    validation_df['data_path']  = config_dict["validation_nifti_dir"] + validation_df[config_dict["data_column_name"]]

    print("Training   Sample:",  len(train_df))
    print("Validation Sample:",  len(validation_df))

    training_images = train_df['data_path'].tolist()
    training_labels = train_df[config_dict["label_column_name"]].tolist()

    validation_images = validation_df['data_path'].tolist()
    validation_labels = validation_df[config_dict["label_column_name"]].tolist()



    train_files = [{"img": img, "label": label} for img, label in zip(training_images,   training_labels)]
    val_files   = [{"img": img, "label": label} for img, label in zip(validation_images, validation_labels)]

    # Define transforms for image
    train_transforms = generate_classification_train_transform(image_key = config_dict["image_key"], label_key = config_dict["label_key"],   img_patch_size = config_dict["img_patch_size"])
    val_transforms   = generate_classification_val_transform(image_key   = config_dict["image_key"], label_key = config_dict["label_key"],   img_patch_size = config_dict["img_patch_size"])

    post_pred  = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=config_dict["num_classes"])])

    # Define dataset, data loader
    persistent_cache = os.path.join(config_dict["cache_root_dir"], "persistent_cache")

    train_ds         = monai.data.PersistentDataset(data=train_files, transform=train_transforms, cache_dir=persistent_cache)
    train_loader     = monai.data.DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=torch.cuda.is_available())

    val_ds           = monai.data.PersistentDataset(data=val_files, transform=val_transforms, cache_dir=persistent_cache)
    val_loader       = monai.data.DataLoader(val_ds, batch_size=args.val_batch_size, num_workers=args.num_worker, pin_memory=torch.cuda.is_available())





    check_ds     = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["label"])



    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model               = get_classification_model(Model_name=config_dict["Model_name"],spatial_dims=config_dict["spatial_dims"],n_input_channels=config_dict["n_input_channels"], num_classes=config_dict["num_classes"],device=device)
    #--- Training Configurations
    labels               = np.array(training_labels)
    class_weights        = compute_class_weights(labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_function        = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)


    #optimizer            = torch.optim.Adam(model.parameters(), 1e-5)

    optimizer            = torch.optim.SGD(model.parameters(),args.lr,momentum=0.9,weight_decay=3e-5,nesterov=True,)
    after_scheduler      = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    scheduler_warmup     = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=after_scheduler)
    scaler               = torch.cuda.amp.GradScaler() if amp else None
    auc_metric           = ROCAUCMetric()
    optimizer.zero_grad()
    optimizer.step()
    #---------- Tripical Training---
    val_interval          = args.val_interval  # do validation every val_interval epochs
    max_epochs            = args.max_epoch
    epoch_len             = len(train_ds) // train_loader.batch_size


   # Checkpoint Loading
    start_epoch = 0
    best_metric_epoch = float('-inf')
    if args.resume_checkpoint is not None and os.path.isfile(args.resume_checkpoint):
        print(f"=> loading checkpoint '{args.resume_checkpoint}'")
        checkpoint            = torch.load(args.resume_checkpoint, map_location=device)
        start_epoch           = checkpoint['epoch']
        best_metric_epoch = checkpoint.get('best_metric_epoch', best_metric_epoch)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{args.resume_checkpoint}' (epoch {checkpoint['epoch']})")
    else:
        print("=> No checkpoint found or provided. Starting training from scratch.")



    # start a typical PyTorch training
    best_metric       = -1
    best_metric_epoch = -1
    writer = SummaryWriter(tfevent_dir)
    for epoch in range(start_epoch, max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step       = 0
        start_time = time.time()
        scheduler_warmup.step()
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        end_time = time.time()
        print(f"Training time: {end_time-start_time}s")
        # save last trained model
        torch.save(model.state_dict(), model_path[:-3] + "_last.pt")
        print("saved last model")

        if (epoch + 1) % val_interval == 0:  # Save every 5 epochs, adjust as needed
            checkpoint_path = model_path[:-3] + "checkpoint_last.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metric': best_metric,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")


        #-- Validation Loop
        val_epoch_len   = len(val_ds) // val_loader.batch_size
        val_step_i      = 0

        if (epoch + 1) % val_interval == 0:
            model.eval()
            start_time = time.time()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_step_i += 1
                    print(f"Validating:{val_step_i}/{val_epoch_len}")
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                if auc_result > best_metric:
                    best_metric = auc_result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_path)
                    print("saved new best metric model")
                    best_model_checkpoint_path = model_path[:-3] + "checkpoint.pth"
                    torch.save({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_metric': best_metric}, best_model_checkpoint_path)
                print("=> Saved new best metric model checkpoint")

                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best AUC: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
                writer.add_scalar("val_auc",      auc_result, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    tain_main()
