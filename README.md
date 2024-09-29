# AI-in-Lung-Health-Benchmarking-Detection-and-Diagnostic-Models-Across-Multiple-CT-Scan-Datasets [![arXiv](https://img.shields.io/badge/arXiv-2405.04605-<color>.svg)](https://arxiv.org/abs/2405.04605)

# Abstract

**BACKGROUND:** Lung cancer's high mortality rate can be mitigated by early detection, which is
increasingly reliant on artificial intelligence (AI) for diagnostic imaging. However, the performance of AI
models is contingent upon the datasets used for their training and validation.

**METHODS:** This study presents the development and validation of AI models for both nodule detection
and cancer classification tasks. For the detection task, two models (DLCSD-mD and LUNA16-mD) were
developed using the Duke Lung Cancer Screening Dataset (DLCSD), which includes over 2,000 CT scans
from 1,613 patients with more than 3,000 annotations. The models were evaluated on internal (DLCSD)
and external datasets, including LUNA16 (601 patients, 1186 nodules) and NLST (969 patients,1192
nodules), using free-response receiver operating characteristic (FROC) analysis and area under the curve
(AUC) metrics. For the classification task, five models were developed and tested: a randomly initialized
3D ResNet50, state-of-the-art open-access models (Genesis and MedNet3D), an enhanced ResNet50
using Strategic Warm-Start++ (SWS++), and a linear classifier analyzing features from the Foundation
Model for Cancer Biomarkers (FMCB). These models were trained to distinguish between benign and
malignant nodules and evaluated using AUC analysis on internal (DLCSD) and external datasets,
including LUNA16 (433 patients, 677 nodules) and NLST (969 patients,1192 nodules).

**RESULTS:** The DLCSD-mD model achieved an AUC of 0.93 (95% CI: 0.91-0.94) on the internal
DLCSD dataset. External validation results were 0.97 (95% CI: 0.96-0.98) on LUNA16 and 0.75 (95%
CI: 0.73-0.76) on NLST. The LUNA16-mD model recorded an AUC of 0.96 (95% CI: 0.95-0.97) on its
native dataset, with AUCs of 0.91 (95% CI: 0.89-0.93) on DLCSD and 0.71 (95% CI: 0.70-0.72) on
NLST. For the classification task, the ResNet50-SWS++ model recorded AUCs of 0.71 (95% CI: 0.61-
0.81) on DLCSD, 0.90 (95% CI: 0.87-0.93) on LUNA16, and 0.81 (95% CI: 0.79-0.82) on NLST. Other
models showed varying performance across datasets, highlighting the importance of diverse model
approaches for robust classification.

**CONCLUSION:** This benchmarking across multiple datasets establishes the DLCSD as a reliable
resource for lung cancer AI research. By making our models and code publicly available, we aim to
accelerate research, enhance reproducibility, and foster collaborative advancements in medical AI.


### Citation Manuscript 

[![arXiv](https://img.shields.io/badge/arXiv-2405.04605-<color>.svg)](https://arxiv.org/abs/2405.04605)


```ruby
@article{tushar2024ai,
  title={AI in Lung Health: Benchmarking Detection and Diagnostic Models Across Multiple CT Scan Datasets},
  author={Tushar, Fakrul Islam and Wang, Avivah and Dahal, Lavsen and Harowicz, Michael R and Lafata, Kyle J and Tailor, Tina D and Lo, Joseph Y},
  journal={arXiv preprint arXiv:2405.04605},
  year={2024}
}
```
```ruby
Tushar, Fakrul Islam, et al. "AI in Lung Health: Benchmarking Detection and Diagnostic Models Across Multiple CT Scan Datasets." arXiv preprint arXiv:2405.04605 (2024).
```

### Citation Dataset- Duke Lung

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13799069.svg)](https://doi.org/10.5281/zenodo.13799069) 
```ruby
A. Wang, F. I. TUSHAR, M. R. Harowicz, K. J. Lafata, T. D. Tailorand J. Y. Lo, “Duke Lung Cancer Screening Dataset 2024”. Zenodo, Mar. 05, 2024. doi: 10.5281/zenodo.13799069.
```



# Datasets

## Duke Lung Cancer Screening Dataset 2024 (DLCS 2024) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13799069.svg)](https://doi.org/10.5281/zenodo.13799069) 

**Background:** Lung cancer risk classification is an increasingly important area of research as low-dose thoracic CT screening programs have become standard of care for patients at high risk for lung cancer. There is limited availability of large, annotated public databases for the training and testing of algorithms for lung nodule classification.

**Methods:** Screening chest CT scans done between January 1, 2015 and June 30, 2021 at Duke University Health System were considered for this study. Efficient nodule annotation was performed semi-automatically by using a publicly available deep learning nodule detection algorithm trained on the LUNA16 dataset to identify initial candidates, which were then accepted based on nodule location in the radiology text report or manually annotated by a medical student and a fellowship-trained cardiothoracic radiologist.

**Results:** The dataset contains 1613 CT volumes with 2487 annotated nodules, selected from a total dataset of 2061 patients, with the remaining data reserved for future testing. Radiologist spot-checking confirmed the semi-automated annotation had an accuracy rate of >90%.

**Conclusions:** The  Duke Lung Cancer Screening Dataset 2024 is the first large dataset for CT screening for lung cancer reflecting the use of current CT technology. This represents a useful resource of lung cancer risk classification research, and the efficient annotation methods described for its creation may be used to generate similar databases for research in the future


## [NLST](https://github.com/fitushar/AI-in-Lung-Health-Benchmarking-Detection-and-Diagnostic-Models-Across-Multiple-CT-Scan-Datasets/tree/main/NLST_Data_Annotations)

With the [National Lung Screening Trial (NLST)](https://www.nejm.org/doi/full/10.1056/NEJMoa1208962), for detection evaluation, we utilized open-access annotations provided by [Mikhael et al.(2023)](https://doi.org/10.1200/JCO.22.01345). We converted over 9,000 2D slice-level bounding box annotations from more than 900 lung cancer patients into 3D representations, resulting in over 1,100 nodule annotations.


To extract 3D annotations from the 2D annotations, we first verified the 2D annotations within the DICOM images. Then, we extracted the `seriesinstanceuid`, `slice_location`, and `slice_number` from the DICOM headers. Subsequently, the image coordinate locations were converted to world coordinates. After verifying these annotations in the corresponding NIFTI images, we concatenated overlapping consecutive 2D annotations of the same lesion across multiple slices into a single 3D annotation. 

The complete code for generating the 3D annotations, along with a visualization script to display these annotations, will be released soon. A preview of the visualization is shown in this [Jupyter Notebook](https://github.com/fitushar/AI-in-Lung-Health-Benchmarking-Detection-and-Diagnostic-Models-Across-Multiple-CT-Scan-Datasets/blob/main/NLST_Data_Annotations/3D_Annotation_Visualizations_NLST.ipynb).

## LUNA16

[LUNA16](https://luna16.grand-challenge.org/), a refined version of the LIDC-IDRI dataset, was utilized for external validation, applying the standard 10-fold cross-validation procedure for lung nodule detection. For cancer diagnosis classification using LUNA16, we followed a labeling scheme from a previous study ([Pai, S. et al. (2024)](https://www.nature.com/articles/s42256-024-00807-9)), which designated nodules with at least one radiologist's indication of malignancy, resulting in 677 labeled nodules. This scheme is referred to as the “Radiologist-Visual Assessed Malignancy Index” (RVAMI).



# Benchmark- Nodule Detection 

The lung cancer (Nodule) detection task is defined as identifying lung nodules within 3D CT scans and localizing them using 3D bounding boxes. To achieve this, we utilized the [MONAI](https://github.com/Project-MONAI/tutorials/tree/main/detection) detection workflow to train and validate 3D detection models based on RetinaNet, enabling straightforward implementation of our benchmark models.

* **DLCSD-mD:** The model developed using the DLCSD development dataset, underwent training for 300 epochs, with validation performed on 20% of the development set to ensure the selection of the best model
* **LUNA16-mD:** The model trained utilizing the official LUNA16 10-fold cross-validation from the [MONAI tutorial documentation](https://github.com/Project-MONAI/tutorials/tree/main/detection).


#### Pre-Processing
All CT volumes were resampled to a standardized resolution of 0.7 × 0.7 × 1.25 mm (x, y, z). The intensity values of the images were clipped between -1000 and 500 HU, and each volume was normalized to have a mean of 0 and a standard deviation of 1. The models were trained using 3D patches of size 192 × 192 × 80 (x, y, z) and a sliding window approach was applied during the prediction phase to cover the entire volume. All models were trained with identical hyperparameters for 300 epochs, and the optimal model was selected based on the lowest validation loss.

#### Evaluation Metrics
The performance of the models was evaluated using the Free-Response Receiver Operating Characteristic (FROC) analysis, which measures sensitivity at various false positive rates (FPRs). The primary performance metric was the average sensitivity at predefined FPRs: 1/8, 1/4, 1/2, 1, 2, 4, and 8 false positives per scan, as outlined in prior studies. Additionally, lesion-level performance was assessed using the Area Under the Receiver Operating Characteristic Curve (AUC) along with a 96% confidence interval (CI).

## DLCSD-mD Run and Example

### | DLCSD-mD 1.1 Data Pre-processing 


### | DLCSD-mD 1.2 training configs and env files

we provided the pre-processed data-split json files, can be found at: **/ct_detection/datasplit_folds/DukeLungRADs_trcv4_fold1.json**, required by the model for train/validation/evaliation.

First please open **"Config_DukeLungRADS_BaseModel_epoch300_patch192x192y80z.json"**, change the values of 

* **"Model_save_path_and_utils":** the dir where the bash, config, result, tfevent_train and trained_modelfolders will be crearted and store.
* **"raw_img_path":**      directory where the resampled images where store.
* **"dataset_info_path"**: directory where the meta data store if needed.
* **"train_cinfig":** training hyper-parameters defined inthis config file.
* **"bash_path":** directory to save the bash file having model running commands


#### Config_DukeLungRADS_BaseModel_epoch300_patch192x192y80z.json
```ruby
{
  "Model_save_path_and_utils": "path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/",
  "raw_img_path"             : "path/to/Data/LungRADS_resample/",
  "dataset_info_path"        : "path/to/ct_detection/dataset_files/",
  "dataset_split_path"       : "path/to/ct_detection/datasplit_folds/",
  "number_of_folds"          : 4,
  "seed"                     : 200,
  "run_prefix"               : "DukeLungRADS_BaseModel_epoch300_patch192x192y80z",
  "split_prefix"             : "DukeLungRADs_trcv4_fold",
  "train_cinfig"             : "path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/training_config.json",
  "bash_path"                : "path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/bash/"
}

```

#### training_config.json

```ruby
{
	"gt_box_mode": "cccwhd",
	"lr": 1e-2,
	"spacing": [0.703125, 0.703125, 1.25],
	"batch_size": 3,
	"patch_size": [192,192,80],
    "val_interval":  5,
    "val_batch_size": 1,
	"val_patch_size": [512,512,208],
	"fg_labels": [0],
	"n_input_channels": 1,
	"spatial_dims": 3,
	"score_thresh": 0.02,
	"nms_thresh": 0.22,
	"returned_layers": [1,2],
	"conv1_t_stride": [2,2,1],
	"max_epoch": 300,
	"base_anchor_shapes": [[6,8,4],[8,6,5],[10,10,6]],
	"balanced_sampler_pos_fraction": 0.3,
	"resume_training": false,
	"resume_checkpoint_path": "",
  "cached_dir":  "/path/to/data/cache/"
}

```

### | DLCSD-mD 1.3 Generating training/Validation Environment and Bash file

```ruby
bash run.sh
```
#### run.sh
```ruby

python3 /path/to/ct_detection/env_main.py --config /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/Config_DukeLungRADS_BaseModel_epoch300_patch192x192y80z.json
python3 /path/to/ct_detection/bash_main_cvit.py --config /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/Config_DukeLungRADS_BaseModel_epoch300_patch192x192y80z.json

```

### | DLCSD-mD 1.4 Training/validation

The model has been trained on cluster using sigularity, runing the created sub file will be initiated training 

**craete a folder for log: /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/bash/slurm_logs/**
```ruby
sbatch run_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1.sub
```
#### run_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1.sub

```ruby
#!/bin/bash

#SBATCH --job-name=CVIT-VNLST_1
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=fakrulislam.tushar@duke.edu
#SBATCH --nodes=1
#SBATCH -w node001
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=/path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/bash/slurm_logs/run_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1_.%j.out
#SBATCH --error=/path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/bash/slurm_logs/run_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1._%j.err

module load singularity/singularity.module
export NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

echo "VNLST Run "
echo "Job Running On "; hostname
echo "Nvidia Visible Devices: $NVIDIA_VISIBLE_DEVICES"

singularity run --nv --bind /path/to /home/ft42/For_Tushar/vnlst_ft42_v1.sif python3 /path/to/ct_detection/training.py -e /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/config/environment_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1.json -c /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/training_config.json
singularity run --nv --bind /path/to /home/ft42/For_Tushar/vnlst_ft42_v1.sif python3 /path/to/ct_detection/testing.py -e /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/config/environment_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1.json -c /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/training_config.json

```

* **you may also chose to use Docker container and simple python call, in that case please check the docker container requiremnt mentioned at [fitushar/Luna16_Monai_Model_XAI_Project](https://github.com/fitushar/Luna16_Monai_Model_XAI_Project)**

```ruby
python3 /path/to/ct_detection/training.py -e /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/config/environment_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1.json -c /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/training_config.json

python3 /path/to/ct_detection/testing.py -e /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/config/environment_DukeLungRADS_BaseModel_epoch300_patch192x192y80z_fold1.json -c /path/to/ct_detection/DukeLungRADS_BaseModel_epoch300_patch192x192y80z/training_config.json
```


### | DLCSD-mD 1.5 Testing 

🚀 Coming Soon

### | DLCSD-mD 1.6 Evaluation and Benchmark


🚀 Coming Soon


### codes

🚀 Coming Soon


# Benchmark- Lungs Cancer Classification Task


We define the lung cancer classification task as given a nodule classifying it as cancer or no-cancer. To benchmark the lung cancer classification task, we employed five different baseline models, including randomly initialized, supervised, and self-supervised pre-trained models, as well as our in-house proposed Strategic Warm-Start++ (SWS++) model.


![Cancer Classification](https://github.com/fitushar/AI-in-Lung-Health-Benchmarking-Detection-and-Diagnostic-Models-Across-Multiple-CT-Scan-Datasets/blob/main/readme_figures/cancer_classifications_1.PNG)

* **3D ResNet50**
* **FMCB:** We adopted a recently published foundational model based on a self-supervised ResNet50, referred to as “FMCB.” We used it to extract 4,096 features per data point and trained a logistic regression model using the scikit-learn framework as suggested by authors. [Pai, S. et al. (2024)](https://www.nature.com/articles/s42256-024-00807-9)
*  **Genesis:**  Models-Genesis's pre-trained ResNet50, added a classification layer on top of it and trained end-to-end. [Zhou, Z., et al. (2021)](https://www.sciencedirect.com/science/article/pii/S1361841520302048)
*  **MedNet3D:** Med3D’s ResNet50 pre-trained ResNet50, we have added a classification layer on top of it and trained end-to-end. [Chen,S., et al. (2019)](https://arxiv.org/abs/1904.00625)
*  **ResNet50-SWS++:** We developed an in-house model using our novel Strategic WarmStart++ (SWS++) pretraining approach. The method involved training a ResNet50 to reduce false positives in lung nodule detection, using a carefully stratified dataset based on nodule confidence scores. The resulting model, “ResNet50-SWS++,” was then fine-tuned for end-to-end lung cancer classification. [Tushar, F. I., et al. (2024)](https://arxiv.org/abs/2405.04605)


## 🚀 Coming Soon
We are currently working on releasing the complete codebase for detection, classification (model weights), pre-processing, and training/validation pipelines. This release will include detailed scripts and instructions to facilitate reproducibility and support further research in this area. 

Stay tuned! The code will be available here soon.

![Coming Soon](https://img.shields.io/badge/Status-Coming%20Soon-orange)




# Citations

* Tushar, Fakrul Islam, et al. "AI in Lung Health: Benchmarking Detection and Diagnostic Models Across Multiple CT Scan Datasets." arXiv preprint arXiv:2405.04605 (2024).
* A. Wang, F. I. TUSHAR, M. R. Harowicz, K. J. Lafata, T. D. Tailorand J. Y. Lo, “Duke Lung Cancer Screening Dataset 2024”. Zenodo, Mar. 05, 2024. doi: 10.5281/zenodo.13799069.
* Mikhael, Peter G., et al. "Sybil: a validated deep learning model to predict future lung cancer risk from a single low-dose chest computed tomography." Journal of Clinical Oncology 41.12 (2023): 2191-2200.
* Pai, S., Bontempi, D., Hadzic, I. et al. Foundation model for cancer imaging biomarkers. Nat Mach Intell 6, 354–367 (2024). https://doi.org/10.1038/s42256-024-00807-9
* Cardoso, M. Jorge, et al. "Monai: An open-source framework for deep learning in healthcare." arXiv preprint arXiv:2211.02701 (2022).
* Z. Zhou, V. Sodha, J. Pang, M. B. Gotway, and J. Liang, "Models genesis," Medical image analysis, vol. 67, p. 101840, 2021.
* S. Chen, K. Ma, and Y. Zheng, "Med3d: Transfer learning for 3d medical image analysis," arXiv preprint arXiv:1904.00625, 2019.
* National Lung Screening Trial Research Team. "Results of initial low-dose computed tomographic screening for lung cancer." New England Journal of Medicine 368.21 (2013): 1980-1991.
* Tushar, Fakrul Islam, et al. "Virtual NLST: towards replicating national lung screening trial." Medical Imaging 2024: Physics of Medical Imaging. Vol. 12925. SPIE, 2024.
* Tushar, Fakrul Islam, et al. "VLST: Virtual Lung Screening Trial for Lung Cancer Detection Using Virtual Imaging Trial." arXiv preprint arXiv:2404.11221 (2024).
