# AI-in-Lung-Health-Benchmarking-Detection-and-Diagnostic-Models-Across-Multiple-CT-Scan-Datasets

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


### Citation
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
