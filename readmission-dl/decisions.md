Decision log

Decision 1: Data cleaning strategy

What I did:
I split the data before preprocessing to avoid leakage, then applied structured cleaning: handled invalid age values (>100), imputed missing glucose using median from training data, applied log transformation to skewed features (length_of_stay_days, creatinine), engineered interaction features (age_los, comorbidity_los), extracted temporal features (month, day), capped outliers using IQR, and applied one-hot encoding with strict column alignment.

Why I did it:
EDA showed skewed medical variables and extreme outliers, which could bias models like XGBoost. Missing glucose values were non-trivial, and median was robust to outliers. Interaction features were added because readmission risk is influenced by combined effects (e.g., age × hospital stay). Column alignment was critical since categorical values differed between splits.

What I considered and rejected:
I considered mean imputation (rejected due to skew), dropping rows with missing values (data loss), and no outlier handling (risk of unstable model behavior).

What would happen if I was wrong here:
Poor preprocessing would propagate noise, causing unstable training, biased predictions, and reduced generalization — especially harming minority class detection.

Decision 2: Model architecture and handling class imbalance

What I did:
I experimented with Deep Learning, Random Forest, and XGBoost. Final model uses XGBoost with class imbalance handled via scale_pos_weight, and an ensemble with deep learning probabilities to improve recall.

Why I did it:
The dataset is tabular with mixed feature types, where tree-based models typically outperform neural networks. However, DL showed higher recall (0.69), which is critical for identifying readmissions. The class imbalance (~90:10) required weighting rather than resampling to avoid overfitting. XGBoost provided stable performance while DL improved sensitivity.

What I considered and rejected:
SMOTE was considered but rejected due to risk of generating unrealistic medical samples. Pure deep learning was also rejected due to instability and no clear performance gain.

What would happen if I was wrong here:
Improper imbalance handling would bias the model toward majority class, missing high-risk patients — which is unacceptable in a healthcare context.

Decision 3: Evaluation metric and threshold selection

What I did:
I used F1-score as the primary metric and tuned the classification threshold instead of relying on the default 0.5. I evaluated precision-recall trade-offs and selected a threshold that improves recall without severely hurting precision.

Why I did it:
Accuracy was misleading due to class imbalance (high accuracy but poor minority detection). F1-score balances precision and recall, making it suitable for this problem. Threshold tuning was necessary because model probabilities were not well-calibrated for the minority class.

What I considered and rejected:
I considered accuracy and ROC-AUC but rejected them as they don’t reflect minority class performance clearly. Fixed threshold (0.5) was also rejected as suboptimal.

What would happen if I was wrong here:
Using incorrect metrics or thresholds would give a false sense of performance, leading to models that fail to detect readmitted patients in real-world scenarios.
