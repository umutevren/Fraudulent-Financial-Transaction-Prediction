import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_features = []
        
    def identify_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numeric and categorical features in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of numeric and categorical feature lists
        """
        # Identify numeric features (Per1-9, Dem1-9, Cred1-6)
        numeric_features = (
            [f"Per{i}" for i in range(1, 10)] +
            [f"Dem{i}" for i in range(1, 10)] +
            [f"Cred{i}" for i in range(1, 7)] +
            ['Normalised_FNT', 'geo_score', 'qsets_normalized_tat', 
             'instance_scores', 'lambda_wt']
        )
        
        # Categorical features
        categorical_features = ['Group']
        
        self.numeric_features = numeric_features
        return numeric_features, categorical_features
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        # For numeric features, fill with median
        for col in self.numeric_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical features, fill with mode
        df['Group'] = df['Group'].fillna(df['Group'].mode()[0])
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different numeric columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional interaction features
        """
        # Create ratios between Per features
        df['Per_ratio_1_2'] = df['Per1'] / (df['Per2'] + 1e-8)
        df['Per_ratio_3_4'] = df['Per3'] / (df['Per4'] + 1e-8)
        
        # Create aggregated features
        df['Cred_mean'] = df[[f'Cred{i}' for i in range(1, 7)]].mean(axis=1)
        df['Dem_mean'] = df[[f'Dem{i}' for i in range(1, 10)]].mean(axis=1)
        df['Per_mean'] = df[[f'Per{i}' for i in range(1, 10)]].mean(axis=1)
        
        # Create risk score
        df['risk_score'] = (df['instance_scores'] * df['geo_score'] * 
                          df['lambda_wt'] * df['qsets_normalized_tat'])
        
        return df
    
    def scale_features(self, train_df: pd.DataFrame, 
                      test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numeric features using StandardScaler.
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            
        Returns:
            Tuple of scaled training and testing DataFrames
        """
        # Fit scaler on training data
        self.scaler.fit(train_df[self.numeric_features])
        
        # Transform both datasets
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()
        
        train_scaled[self.numeric_features] = self.scaler.transform(train_df[self.numeric_features])
        test_scaled[self.numeric_features] = self.scaler.transform(test_df[self.numeric_features])
        
        return train_scaled, test_scaled
    
    def prepare_features(self, train_df: pd.DataFrame, 
                        test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main function to prepare features for both training and test sets.
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            
        Returns:
            Tuple of processed training and testing DataFrames
        """
        logger.info("Starting feature engineering process...")
        
        # Save original data
        os.makedirs('data/processed', exist_ok=True)
        train_df.to_csv('data/processed/train_before_engineering.csv', index=False)
        test_df.to_csv('data/processed/test_before_engineering.csv', index=False)
        
        # Identify features
        self.identify_features(train_df)
        
        # Handle missing values
        train_df = self.handle_missing_values(train_df)
        test_df = self.handle_missing_values(test_df)
        
        # Create new features
        train_df = self.create_interaction_features(train_df)
        test_df = self.create_interaction_features(test_df)
        
        # Save intermediate features
        train_df.to_csv('data/processed/train_with_interactions.csv', index=False)
        test_df.to_csv('data/processed/test_with_interactions.csv', index=False)
        
        # Scale features
        train_scaled, test_scaled = self.scale_features(train_df, test_df)
        
        # Save final processed features
        train_scaled.to_csv('data/processed/train_final.csv', index=False)
        test_scaled.to_csv('data/processed/test_final.csv', index=False)
        logger.info("Feature engineering completed successfully")
        
        return train_scaled, test_scaled 