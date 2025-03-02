import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict, Any
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                class_weight='balanced'
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=random_state,
                scale_pos_weight=10  # Adjust based on class imbalance
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=random_state,
                class_weight='balanced'
            )
        }
        self.best_model = None
        self.best_model_name = None
        
    def prepare_training_data(self, X: pd.DataFrame, y: pd.Series,
                            test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and validation datasets with SMOTE oversampling.
        
        Args:
            X: Feature DataFrame
            y: Target series
            test_size: Validation set size
            
        Returns:
            Tuple of X_train, X_val, y_train, y_val
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Count samples in minority class
        n_minority = sum(y_train == 1)
        
        # Adjust k_neighbors based on the number of minority samples
        k_neighbors = min(5, n_minority - 1)
        
        # Apply SMOTE to training data only with adjusted parameters
        smote = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"Training data shape after SMOTE: {X_train_resampled.shape}")
        return X_train_resampled, X_val, y_train_resampled, y_val
    
    def train_and_evaluate(self, X_train: np.ndarray, X_val: np.ndarray,
                          y_train: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            
        Returns:
            Dictionary with model performances
        """
        results = {}
        best_auc = 0.0
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = {
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'auc_roc': roc_auc_score(y_val, y_pred_proba)
            }
            
            results[name] = metrics
            
            # Update best model
            if metrics['auc_roc'] > best_auc:
                best_auc = metrics['auc_roc']
                self.best_model = model
                self.best_model_name = name
            
            logger.info(f"{name} metrics: {metrics}")
        
        logger.info(f"Best model: {self.best_model_name} with AUC-ROC: {best_auc:.4f}")
        return results
    
    def save_model(self, output_dir: str = 'models'):
        """
        Save the best model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = Path(output_dir) / f"{self.best_model_name}_best_model.joblib"
        joblib.dump(self.best_model, model_path)
        logger.info(f"Best model saved to {model_path}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using the best model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probability predictions
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        return self.best_model.predict_proba(X)[:, 1] 