import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define dtypes for efficiency
TRAIN_DTYPES = {
    'id': 'int32',
    'Group': 'category',
    'Target': 'int8',
    'Per1': 'float32',
    'Per2': 'float32',
    'Per3': 'float32',
    'Per4': 'float32',
    'Per5': 'float32',
    'Per6': 'float32',
    'Per7': 'float32',
    'Per8': 'float32',
    'Per9': 'float32',
    'Dem1': 'float32',
    'Dem2': 'float32',
    'Dem3': 'float32',
    'Dem4': 'float32',
    'Dem5': 'float32',
    'Dem6': 'float32',
    'Dem7': 'float32',
    'Dem8': 'float32',
    'Dem9': 'float32',
    'Cred1': 'float32',
    'Cred2': 'float32',
    'Cred3': 'float32',
    'Cred4': 'float32',
    'Cred5': 'float32',
    'Cred6': 'float32',
    'Normalised_FNT': 'float32'
}

# Test data has the same dtypes except no Target column
TEST_DTYPES = {k: v for k, v in TRAIN_DTYPES.items() if k != 'Target'}

# Additional features dtypes
GEO_DTYPES = {'id': 'int32', 'geo_score': 'float32'}
LAMBDA_DTYPES = {'Group': 'category', 'lambda_wt': 'float32'}
QSET_DTYPES = {'id': 'int32', 'qsets_normalized_tat': 'float32'}
INSTANCE_DTYPES = {'id': 'int32', 'instance_scores': 'float32'}

def load_data_with_dtypes(file_path, dtypes=None, nrows=None, usecols=None):
    """Load data with optimized dtypes to reduce memory usage."""
    try:
        df = pd.read_csv(file_path, dtype=dtypes, nrows=nrows, usecols=usecols)
        logger.info(f"Successfully loaded {file_path} with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise

def optimize_dtypes(df):
    """Optimize datatypes to reduce memory usage."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def prepare_datasets(save_processed=True):
    """Load and prepare all datasets with optimized memory usage."""
    
    # Define dtypes and columns for main datasets
    float_cols = ['Per1', 'Per2', 'Per3', 'Per4', 'Per5', 'Per6', 'Per7', 'Per8', 'Per9',
                 'Dem1', 'Dem2', 'Dem3', 'Dem4', 'Dem5', 'Dem6', 'Dem7', 'Dem8', 'Dem9',
                 'Cred1', 'Cred2', 'Cred3', 'Cred4', 'Cred5', 'Cred6', 'Normalised_FNT']
    
    dtypes = {col: 'float32' for col in float_cols}
    dtypes.update({
        'id': 'int32',
        'Group': 'category',
        'Target': 'int8'
    })

    # Load main datasets
    logger.info("Loading train dataset...")
    train_df = load_data_with_dtypes('data/raw/train.csv', dtypes=dtypes)
    
    logger.info("Loading test dataset...")
    test_df = load_data_with_dtypes('data/raw/test_share.csv', dtypes=dtypes)

    # Get unique IDs from train and test for filtering
    all_ids = pd.concat([train_df['id'], test_df['id']]).unique()
    logger.info(f"Total unique IDs to process: {len(all_ids)}")

    # Load and merge additional features efficiently
    logger.info("Loading and merging additional features...")
    
    # Load only necessary columns and rows for Geo scores
    logger.info("Processing Geo scores...")
    geo_scores = load_data_with_dtypes(
        'data/raw/Geo_scores.csv',
        dtypes={'id': 'int32', 'geo_score': 'float32'},
        usecols=['id', 'geo_score']
    )
    geo_scores = geo_scores[geo_scores['id'].isin(all_ids)]
    
    # Merge Geo scores
    train_df = pd.merge(train_df, geo_scores[['id', 'geo_score']], on='id', how='left')
    test_df = pd.merge(test_df, geo_scores[['id', 'geo_score']], on='id', how='left')
    del geo_scores
    gc.collect()
    
    # Load and merge Q-set TATs
    logger.info("Processing Q-set TATs...")
    qset_tats = load_data_with_dtypes(
        'data/raw/Qset_tats.csv',
        dtypes={'id': 'int32', 'qsets_normalized_tat': 'float32'},
        usecols=['id', 'qsets_normalized_tat']
    )
    qset_tats = qset_tats[qset_tats['id'].isin(all_ids)]
    
    train_df = pd.merge(train_df, qset_tats[['id', 'qsets_normalized_tat']], on='id', how='left')
    test_df = pd.merge(test_df, qset_tats[['id', 'qsets_normalized_tat']], on='id', how='left')
    del qset_tats
    gc.collect()
    
    # Load and merge instance scores
    logger.info("Processing instance scores...")
    instance_scores = load_data_with_dtypes(
        'data/raw/instance_scores.csv',
        dtypes={'id': 'int32', 'instance_scores': 'float32'},
        usecols=['id', 'instance_scores']
    )
    instance_scores = instance_scores[instance_scores['id'].isin(all_ids)]
    
    train_df = pd.merge(train_df, instance_scores[['id', 'instance_scores']], on='id', how='left')
    test_df = pd.merge(test_df, instance_scores[['id', 'instance_scores']], on='id', how='left')
    del instance_scores
    gc.collect()
    
    # Load and merge lambda weights
    logger.info("Processing lambda weights...")
    lambda_wts = load_data_with_dtypes(
        'data/raw/Lambda_wts.csv',
        dtypes={'Group': 'category', 'lambda_wt': 'float32'}
    )
    
    train_df = pd.merge(train_df, lambda_wts[['Group', 'lambda_wt']], on='Group', how='left')
    test_df = pd.merge(test_df, lambda_wts[['Group', 'lambda_wt']], on='Group', how='left')
    del lambda_wts
    gc.collect()

    # Save processed datasets if requested
    if save_processed:
        logger.info("Saving processed datasets...")
        os.makedirs('data/processed', exist_ok=True)
        
        logger.info("Saving train dataset in parquet format...")
        train_df.to_parquet('data/processed/train_processed.parquet', index=False, compression='snappy')
        
        logger.info("Saving test dataset in parquet format...")
        test_df.to_parquet('data/processed/test_processed.parquet', index=False, compression='snappy')
        
        logger.info("Processed datasets saved successfully")

    return train_df, test_df

def merge_features(df, geo_scores, lambda_wts, qset_tats, instance_scores):
    """Merge all features efficiently."""
    logger.info("Starting optimized feature merging process...")
    
    # Sort DataFrames by merge keys for better performance
    logger.info("Sorting DataFrames by merge keys...")
    df = df.sort_values('id')
    geo_scores = geo_scores.sort_values('id')
    qset_tats = qset_tats.sort_values('id')
    instance_scores = instance_scores.sort_values('id')
    lambda_wts = lambda_wts.sort_values('Group')

    # Merge id-based features
    logger.info("Merging id-based features...")
    df = pd.merge(df, geo_scores[['id', 'geo_score']], on='id', how='left')
    df = pd.merge(df, qset_tats[['id', 'qsets_normalized_tat']], on='id', how='left')
    df = pd.merge(df, instance_scores[['id', 'instance_scores']], on='id', how='left')
    
    # Merge Group-based features
    df = pd.merge(df, lambda_wts[['Group', 'lambda_wt']], on='Group', how='left')

    logger.info(f"Final merged dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    train_df, test_df = prepare_datasets(save_processed=True) 