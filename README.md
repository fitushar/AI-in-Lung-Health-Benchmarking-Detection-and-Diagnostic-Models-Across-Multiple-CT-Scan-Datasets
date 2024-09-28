# AI-in-Lung-Health-Benchmarking-Detection-and-Diagnostic-Models-Across-Multiple-CT-Scan-Datasets

# Abstract : https://doi.org/10.48550/arXiv.2405.04605

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

## 🚀 Coming Soon
We are currently working on releasing the complete codebase for detection, classification (model weights), pre-processing, and training/validation pipelines. This release will include detailed scripts and instructions to facilitate reproducibility and support further research in this area. 

Stay tuned! The code will be available here soon.

![Coming Soon](https://img.shields.io/badge/Status-Coming%20Soon-orange)




# Citations

* Tushar, Fakrul Islam, et al. "AI in Lung Health: Benchmarking Detection and Diagnostic Models Across Multiple CT Scan Datasets." arXiv preprint arXiv:2405.04605 (2024).
* A. Wang, F. I. TUSHAR, M. R. Harowicz, K. J. Lafata, T. D. Tailorand J. Y. Lo, “Duke Lung Cancer Screening Dataset 2024”. Zenodo, Mar. 05, 2024. doi: 10.5281/zenodo.13799069.
* Mikhael, Peter G., et al. "Sybil: a validated deep learning model to predict future lung cancer risk from a single low-dose chest computed tomography." Journal of Clinical Oncology 41.12 (2023): 2191-2200.
* National Lung Screening Trial Research Team. "Results of initial low-dose computed tomographic screening for lung cancer." New England Journal of Medicine 368.21 (2013): 1980-1991.
* Tushar, Fakrul Islam, et al. "Virtual NLST: towards replicating national lung screening trial." Medical Imaging 2024: Physics of Medical Imaging. Vol. 12925. SPIE, 2024.
* Tushar, Fakrul Islam, et al. "VLST: Virtual Lung Screening Trial for Lung Cancer Detection Using Virtual Imaging Trial." arXiv preprint arXiv:2404.11221 (2024).
