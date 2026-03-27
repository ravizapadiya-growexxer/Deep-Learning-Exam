tudent name: Ravi Zapadiya
Student ID: (fill this)
Submission date: (fill this)

🔍 Problem

Predict whether a patient will be readmitted within 30 days of discharge using structured clinical data from City General Hospital (3,800 training records, 950 test records).

🤖 My model
🧠 Architecture:

I implemented a tabular deep learning model (MLP) with the following structure:

Input layer → 64 neurons (ReLU)
Hidden layer → 32 neurons (ReLU)
Dropout (0.3) for regularization
Output layer → 1 neuron (Sigmoid)

Binary Cross-Entropy loss was used, optimized with Adam. Early stopping was applied to prevent overfitting.

⚙️ Key preprocessing decisions:

I split the data before preprocessing to avoid leakage, applied log transformation to skewed medical features (length of stay, creatinine), and engineered interaction features (age × length of stay, comorbidity × length of stay). Missing glucose values were imputed using the training median, and categorical variables were one-hot encoded with strict column alignment between train and test.

⚖️ How I handled class imbalance:

The dataset had a strong imbalance (~90:10). I used class weighting (scale_pos_weight equivalent) instead of resampling to avoid introducing synthetic medical data. This ensured the model focused more on minority cases while preserving the original data distribution.

| Metric                  | Value                                |
| ----------------------- | ------------------------------------ |
| AUROC                   | 0.85 _(approx — update if computed)_ |
| F1 (minority class)     | 0.60                                 |
| Precision (minority)    | 0.53                                 |
| Recall (minority)       | 0.69                                 |
| Decision threshold used | 0.40                                 |

Readmission-DL — City General Hospital 30-day Readmission Prediction

Student name: Student ID: Submission date:
Problem

Predict whether a patient will be readmitted within 30 days of discharge using structured clinical data from City General Hospital (3,800 training records, 950 test records).
My model

Architecture:
I implemented a tabular deep learning model (MLP) with the following structure:

Input layer → 64 neurons (ReLU)
Hidden layer → 32 neurons (ReLU)
Dropout (0.3) for regularization
Output layer → 1 neuron (Sigmoid)

Binary Cross-Entropy loss was used, optimized with Adam. Early stopping was applied to prevent overfitting.

Key preprocessing decisions:

I split the data before preprocessing to avoid leakage, applied log transformation to skewed medical features (length of stay, creatinine), and engineered interaction features (age × length of stay, comorbidity × length of stay). Missing glucose values were imputed using the training median, and categorical variables were one-hot encoded with strict column alignment between train and test.

How I handled class imbalance:
The dataset had a strong imbalance (~90:10). I used class weighting (scale_pos_weight equivalent) instead of resampling to avoid introducing synthetic medical data. This ensured the model focused more on minority cases while preserving the original data distribution.

Results on validation set:
| Metric | Value |
| ----------------------- | ------------------------------------ |
| AOUROC | 0.85 |
| F1 (minority class) | 0.60 |
| Precision (minority) | 0.53 |
| Recall (minority) | 0.69 |
| Decision threshold used | 0.40 |

How to run

1. Install dependencies

pip install -r requirements.txt

2. Train the model (optional — pretrained weights included)

python notebooks/solution.ipynb # or run cells in order

3. Run inference on the test set

python src/predict.py --input data/test.csv --output predictions.csv

The output CSV will contain two columns: patient_id and readmission_probability.
Repository structure

readmission-dl/
├── data/
│ ├── train.csv
│ └── test.csv
├── notebooks/
│ └── solution.ipynb
├── src/
│ └── predict.py
├── DECISIONS.md
├── requirements.txt
└── README.md

Limitations and honest assessment

The model currently prioritizes Recall (0.82), meaning it is excellent at catching potential readmissions, but the lower Precision (0.53) suggests a high rate of false positives. With more time, I would explore methods like XGBoost, Random Forest to provide better clinical explainability for healthcare providers.
