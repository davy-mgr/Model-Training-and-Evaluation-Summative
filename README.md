# Model-Training-and-Evaluation-Summative

# Predicting 30-Day Hospital Readmission in Diabetic Patients: Traditional ML vs. Deep Learning

Summative project for the Machine Learning module. The project compares Scikit-Learn models and Keras
neural networks (Sequential and Functional APIs with `tf.data`) on the UCI *Diabetes 130-US Hospitals
(1999-2008)* dataset.

## Links
- Report: _add link_
- Google Drive (raw dataset zip + saved models): _add link_
- Presentation video: _add link_

## Problem
Predict whether a diabetic inpatient is readmitted `<30` days, `>30` days, or not at all (3-class), plus a
binary follow-up (`<30` vs. rest). About 101k encounters, ~50 raw features.

## Key design choices
- **Patient-grouped splitting** (`StratifiedGroupKFold` on `patient_nbr`) to prevent patient leakage, with an
  experiment quantifying the leakage effect.
- Removal of expired/hospice discharges, which cannot be readmitted.
- Feature engineering: ICD-9 grouping, medication-change counts, prior-utilisation totals, intensity ratios,
  testing flags. An ablation measures their value.
- Traditional ML: Logistic Regression, Random Forest, Histogram Gradient Boosting, SVM (subsampled), tuned with
  grouped-CV `RandomizedSearchCV`.
- Deep learning: Keras Sequential and Functional (shared ICD-9 embedding) models, `tf.data` input pipeline,
  random hyperparameter search, early stopping.
- Experiment tracking: every run gets an ID (E01, E02, ...) and is logged to **TensorBoard** (`tb_logs/`):
  hyperparameters (HParams), metrics, training time, per-epoch curves for Keras runs and key figures. The
  results table is exported to `results/experiment_table.csv` and every row maps to a TensorBoard run.
- Evaluation: learning curves, confusion matrices, OvR ROC-AUC, PR curves, error and subgroup analysis.

## Reproduce
1. `pip install -r requirements.txt`
2. Open `readmission_ml_dl.ipynb` in Jupyter or Colab and run all cells. The notebook downloads the dataset
   from the UCI repository into `data/`. If the download fails, get the zip from the Drive folder above and
   extract it into `data/`.
3. Everything is seeded (`SEED = 42`). Runtime knobs (`N_DL_TRIALS`, `N_ML_SEARCH_ITER`, `SVM_SUBSAMPLE`)
   are in the config cell.

## View the TensorBoard logs
```
tensorboard --logdir tb_logs
```
In Colab/Jupyter, the launcher cell in Section 11 of the notebook opens it inline.

## Repository layout
```
readmission_ml_dl.ipynb   main notebook
requirements.txt
README.md
figures/                  generated plots (optional to commit)
tb_logs/                  TensorBoard logs backing the experiment table (commit these)
results/                  experiment_table.csv
```
`data/` and `models/` are git-ignored. Raw data and saved models (`best_ml_pipeline.joblib`,
`best_dl_model.keras`, `dl_preprocessing.joblib`) are on Google Drive.

## Data citation
Strack, B. et al. (2014). Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000
clinical database patient records. *BioMed Research International*. Dataset: UCI Machine Learning Repository.
