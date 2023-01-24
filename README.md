
# Feature relevance in Anomaly Detection

This repository contains the showcases conducted in the paper "Feature relevance XAI in anomaly detection:
reviewing approaches and challenges"


## Showcases

### MVTec 

#### Data
MVTec data is available at https://www.mvtec.com/company/research/datasets/mvtec-ad/ or https://www.kaggle.com/datasets/ipythonx/mvtec-ad
To replicate the MVTec results of the paper, download the data and copy the `grid` dataset folder into `data/MVTec/`.

[Bergmann, Paul, et al. "The MVTec anomaly detection dataset: a comprehensive real-world dataset for unsupervised anomaly detection." International Journal of Computer Vision 129.4 (2021): 1038-1059.]

#### Anomaly Detector
Student-teacher network for anomaly segmentation on MVTec data. 

[Wang, Guodong, et al. "Student-teacher feature pyramid matching for unsupervised anomaly detection." arXiv preprint arXiv:2103.04257 (2021).]

Code adapted from the unofficial implementation of xiahaifeng1995 https://github.com/xiahaifeng1995/STPM-Anomaly-Detection-Localization-master


#### Running Showcases
`mvtec_nn_xai.py` contains the nearest neighbor implementation of SHAP. 
All remaining MVTec showcases are contained within `mvtec_xai.py`.

The trained model used in the review paper and the achieved explanations are available at: https://professor-x.de/feature-relevance-AD-results

### ERP

#### Data
ERP data is available at https://professor-x.de/erp-fraud-data

[Tritscher, Julian, et al. "Open ERP System Data For Occupational Fraud Detection." arXiv preprint arXiv:2206.04460 (2022).]

#### Anomaly Detector
Autoencoder neural network with hyperparameters taken from 

[Tritscher, Julian, et al. "Towards explainable occupational fraud detection." In Workshop on Mining Data for Financial Applications (Springer) (2022).]

Code adapted from the official implementation: https://github.com/LSX-UniWue/explainable-ERP-fraud-detection

#### Running Showcases
All ERP showcases are contained within `run_xai.py`.

*Note*: Additional setup is required for running SHAP with optimized Takeishi reference data.
To integrate the optimization procedure directly within kernel-SHAP, 
this implementation requires to manually override the `shap/explainer/_kernel.py` script within the SHAP package.
For this, either override the contents of `shap/explainer/_kernel.py` entirely 
with the backup file provided in `xai/backups/shap_kernel_backup.py`
or add the small segments marked with `# NEWCODE` within `xai/backups/shap_kernel_backup.py` in the 
original library file of `shap/explainer/_kernel.py`.

The trained model and hyperparameters used in the review paper are available in `outputs/models/`.
The achieved explanations are available at: https://professor-x.de/feature-relevance-AD-results


