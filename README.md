# Fraud Detection Pipeline 🕵️‍♂️

After spending countless hours dealing with imbalanced fraud datasets and struggling with model performance, I decided to build a robust fraud detection pipeline that actually works. This project is the result of real-world experience and countless iterations to get things right.

## 📊 Data Source

This project uses the [Fraudulent Financial Transaction Prediction dataset](https://www.kaggle.com/datasets/younusmohamed/fraudulent-financial-transaction-prediction/data) from Kaggle, created by Younus Mohamed. The dataset includes:

- Main transaction data (`train.csv`, `test_share.csv`)
- Geographical risk scores (`Geo_scores.csv`)
- Group-based lambda weights (`Lambda_wts.csv`)
- Q-set timing analysis (`Qset_tats.csv`)
- Instance-based scoring (`instance_scores.csv`)

A huge thank you to Younus Mohamed for providing this comprehensive dataset for fraud detection research.

## 🌟 Key Features

- **Smart Data Loading**: Optimized data loading with proper dtypes (saved about 40% memory!)
- **Advanced Feature Engineering**: Created meaningful interaction features that capture fraud patterns
- **Sophisticated Model Ensemble**: Combination of Random Forest, XGBoost, and LightGBM (because why choose one when you can have the best of all?)
- **Beautiful Visualizations**: Comprehensive visual analysis at every step of the pipeline

## 🛠 Technical Deep Dive

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

### Feature Engineering Magic ✨

The feature engineering pipeline includes:
- **Interaction Features**: Created ratios between personal features (Per1/Per2, Per3/Per4) that help capture unusual patterns
- **Aggregated Metrics**: Computed mean values across feature groups (Credit, Demographics, Personal)
- **Risk Scoring**: Combined multiple risk factors (geo_score * instance_score * lambda_weight) to create a composite risk indicator
- **Missing Value Treatment**: Smart handling of missing values using medians for numeric features and mode for categorical ones

### Dealing with Imbalanced Data 🎯

One of the biggest challenges in fraud detection is the heavily imbalanced nature of the data. I used SMOTE (Synthetic Minority Over-sampling Technique) with a twist:

```python
# Dynamically adjust SMOTE's k_neighbors based on minority class size
k_neighbors = min(5, n_minority - 1)
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
```

This adaptive approach ensures SMOTE works well even with very few fraud cases, avoiding the common pitfall of having too many neighbors for a small minority class.

### Model Ensemble 🤖

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

### Visualization Suite 📊

I've created a comprehensive visualization module that helps understand:
- Feature distributions with fraud overlay
- Correlation patterns between features
- Model performance comparisons (ROC curves, Precision-Recall curves)
- Feature importance analysis
- Confusion matrices with percentage views

## 📈 Results

The current best model (LightGBM) achieves:
- AUC-ROC: 0.9993
- Precision: 0.67
- Recall: 1.0
- F1 Score: 0.80

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fraud-detection.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/younusmohamed/fraudulent-financial-transaction-prediction/data) and place the files in `data/raw/`

4. Run the pipeline:
```bash
python src/main.py
```

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

## 🤔 Future Improvements

- [ ] Add support for online learning to adapt to new fraud patterns
- [ ] Implement feature selection based on mutual information
- [ ] Add explainability using SHAP values
- [ ] Create an API endpoint for real-time fraud detection

## 📫 Get in Touch

Found a bug? Have a suggestion? Feel free to open an issue or reach out directly. I'm always excited to discuss fraud detection and machine learning!

## 🙏 Acknowledgments

Special thanks to Younus Mohamed for providing the comprehensive fraud detection dataset on Kaggle that made this project possible.

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

## Dataset Description

The project uses the following datasets:

1. `train.csv`: Main training dataset with target variable
2. `test_share.csv`: Test dataset for predictions
3. `Geo_scores.csv`: Geospatial location scores
4. `Lambda_wts.csv`: Group-based weights
5. `Qset_tats.csv`: Network turn-around times
6. `instance_scores.csv`: Risk qualification scores

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fraud-detection
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place all data files in the project root directory:
   - train.csv
   - test_share.csv
   - Geo_scores.csv
   - Lambda_wts.csv
   - Qset_tats.csv
   - instance_scores.csv

2. Run the main script:
```bash
python src/main.py
```

The script will:
- Load and preprocess the data
- Perform feature engineering
- Train multiple models
- Generate and save predictions
- Save the best performing model

## Output

The pipeline generates:
1. Predictions file (`output/predictions.csv`) containing:
   - ID
   - Fraud prediction (0/1)
   - Probability scores

2. Best model file (`models/best_model.joblib`)

## Model Performance

The pipeline evaluates models using:
- Precision: Accuracy of positive predictions
- Recall: Ability to find all positive cases
- F1-score: Harmonic mean of precision and recall
- ROC-AUC: Area under the ROC curve

Performance metrics are logged during execution.

## Feature Engineering

The pipeline implements several feature engineering techniques:
- Handling missing values
- Feature scaling
- Creating interaction features
- Aggregating features by groups
- Computing risk scores

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Fraud Detection Dataset README

This repository contains data for a **Fraud Detection** problem where only a small percentage of transactions are fraudulent. Below are the details of each file and instructions to combine them.

---

## File Descriptions

1. **train.csv**  
   - **Columns** (28 total):  
     - `id` (int) — Unique identifier (masked).  
     - `Group` (string) — Grouping label (masked).  
     - `Per1` to `Per9` (float) — Numeric features (masked).  
     - `Dem1` to `Dem9` (float) — Additional numeric features (masked).  
     - `Cred1` to `Cred6` (float) — Credit/risk features (masked).  
     - `Normalised_FNT` (float) — A numeric field (masked).  
     - `Target` (int) — Fraud indicator (1 = Fraud, 0 = Clean).  

2. **test_share.csv**  
   - **Columns** (27 total): Same as `train.csv` except **no** `Target` column.  
   - Used for final predictions or model evaluation.

3. **Geo_scores.csv**  
   - **Columns**:  
     - `id` (int)  
     - `geo_score` (float)  
   - Contains geospatial location scores related to transactions.

4. **Lambda_wts.csv**  
   - **Columns**:  
     - `Group` (string)  
     - `lambda_wt` (float)  
   - Proprietary weight/score for each group.

5. **Qset_tats.csv**  
   - **Columns**:  
     - `id` (int)  
     - `qsets_normalized_tat` (float)  
   - Network turn-around times (TAT) for each transaction.

6. **instance_scores.csv**  
   - **Columns**:  
     - `id` (int)  
     - `instance_scores` (float)  
   - Vulnerability or risk qualification scores.

---

## Usage

1. **Combine Files**  
   - Merge `Geo_scores.csv`, `Qset_tats.csv`, `instance_scores.csv` on `id`.  
   - Merge `Lambda_wts.csv` on `Group`.  

2. **Model Building**  
   - Explore the imbalance (majority transactions are clean).  
   - Use oversampling (RandomOverSampler) or undersampling or SMOTE to address the minority class.  
   - Train classifiers such as RandomForest, XGBoost, or LightGBM.

3. **Feature Engineering**  
   - Check which additional scores (`geo_score`, `lambda_wt`, etc.) add predictive value.  
   - Consider outlier treatment or scaling if needed.

4. **Evaluation**  
   - Since data is highly imbalanced, use **Precision**, **Recall**, **F1-score**, or **ROC-AUC**.  
   - Do not rely solely on accuracy.

5. **Predicting**  
   - After training, make predictions on `test_share.csv` (which lacks the `Target` column).  
   - Submit results or finalize them as needed.

---

## Example Python Workflow

```python
import pandas as pd

# Read main files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_share.csv')

# Read extra data
geo_df = pd.read_csv('Geo_scores.csv')
lambda_df = pd.read_csv('Lambda_wts.csv')
tat_df = pd.read_csv('Qset_tats.csv')
inst_df = pd.read_csv('instance_scores.csv')

# Merge additional data to train_df if needed
train_merged = pd.merge(train_df, geo_df, on='id', how='left')
train_merged = pd.merge(train_merged, tat_df, on='id', how='left')
train_merged = pd.merge(train_merged, inst_df, on='id', how='left')
# Merge lambda on Group
train_merged = pd.merge(train_merged, lambda_df, on='Group', how='left')

# Proceed with preprocessing, modeling, etc.
