# Fraud Detection Pipeline 🕵

After spending countless hours dealing with imbalanced fraud datasets and struggling with model performance, I decided to build a robust fraud detection pipeline that actually works. This project is the result of real-world experience and countless iterations to get things right.

## Data Source

This project uses the [Fraudulent Financial Transaction Prediction dataset](https://www.kaggle.com/datasets/younusmohamed/fraudulent-financial-transaction-prediction/data) from Kaggle, created by Younus Mohamed. The dataset includes:

- Main transaction data (`train.csv`, `test_share.csv`)
- Geographical risk scores (`Geo_scores.csv`)
- Group-based lambda weights (`Lambda_wts.csv`)
- Q-set timing analysis (`Qset_tats.csv`)
- Instance-based scoring (`instance_scores.csv`)

A huge thank you to Younus Mohamed for providing this comprehensive dataset for fraud detection research.

##  Key Features

- **Smart Data Loading**: Optimized data loading with proper dtypes (saved about 40% memory!)
- **Advanced Feature Engineering**: Created meaningful interaction features that capture fraud patterns
- **Sophisticated Model Ensemble**: Combination of Random Forest, XGBoost, and LightGBM (because why choose one when you can have the best of all?)
- **Beautiful Visualizations**: Comprehensive visual analysis at every step of the pipeline

## Technical Deep Dive

### Data Preprocessing

The data loader handles multiple data sources efficiently:
- Main transaction data (89MB) with 27 base features
- Geographical risk scores (18MB) for location-based risk assessment
- Lambda weights (18KB) for different group categories
- Q-set timing analysis (18MB) for behavioral patterns
- Instance-based scoring (25MB) for transaction-specific risks

Each dataset is loaded with optimized dtypes (e.g., using 'float32' instead of 'float64', 'int32' for IDs) to keep memory usage in check. The preprocessing pipeline includes:

```python
TRAIN_DTYPES = {
    'id': 'int32',
    'Group': 'category',
    'Target': 'int8',
    'Per1': 'float32',
    # ... and so on
}
```

### Feature Engineering Magic 

The feature engineering pipeline includes:
- **Interaction Features**: Created ratios between personal features (Per1/Per2, Per3/Per4) that help capture unusual patterns
- **Aggregated Metrics**: Computed mean values across feature groups (Credit, Demographics, Personal)
- **Risk Scoring**: Combined multiple risk factors (geo_score * instance_score * lambda_weight) to create a composite risk indicator
- **Missing Value Treatment**: Smart handling of missing values using medians for numeric features and mode for categorical ones

### Dealing with Imbalanced Data 

One of the biggest challenges in fraud detection is the heavily imbalanced nature of the data. I used SMOTE (Synthetic Minority Over-sampling Technique) with a twist:

```python
# Dynamically adjust SMOTE's k_neighbors based on minority class size
k_neighbors = min(5, n_minority - 1)
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
```

This adaptive approach ensures SMOTE works well even with very few fraud cases, avoiding the common pitfall of having too many neighbors for a small minority class.

### Model Ensemble 

The pipeline uses three powerful models, each bringing its strengths:

1. **Random Forest**: Great for handling non-linear relationships and categorical features
   - Balanced class weights
   - 100 trees with max depth of 10

2. **XGBoost**: Excellent at finding complex patterns
   - Scale positive weight adjustment for imbalanced data
   - Learning rate of 0.1 with 100 boosting rounds

3. **LightGBM**: Fast and memory-efficient
   - Balanced class weights
   - Leaf-wise growth strategy

### Visualization Suite 

I've created a comprehensive visualization module that helps understand:
- Feature distributions with fraud overlay
- Correlation patterns between features
- Model performance comparisons (ROC curves, Precision-Recall curves)
- Feature importance analysis
- Confusion matrices with percentage views

## Results

The current best model (LightGBM) achieves:
- AUC-ROC: 0.9993
- Precision: 0.67
- Recall: 1.0
- F1 Score: 0.80



## 📁 Project Structure

```
.
├── data/
│   ├── raw/           # Original data files from Kaggle
│   └── processed/     # Processed and engineered features
├── src/
│   ├── data/          # Data loading and preprocessing
│   ├── features/      # Feature engineering pipeline
│   ├── models/        # Model training and evaluation
│   └── visualization/ # Visualization utilities
├── output/
│   └── plots/         # Generated visualizations
└── models/            # Saved model artifacts
```



## Features

- Data loading and merging from multiple sources
- Feature engineering and preprocessing
- Model training with multiple algorithms:
  - Random Forest
  - XGBoost
  - LightGBM
- SMOTE oversampling for handling class imbalance
- Model evaluation using multiple metrics:
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Prediction generation for test data










