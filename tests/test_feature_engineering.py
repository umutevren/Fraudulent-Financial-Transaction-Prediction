import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureEngineer

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(100),
        'Group': ['A', 'B'] * 50,
        'Per1': np.random.randn(100),
        'Per2': np.random.randn(100),
        'Dem1': np.random.randn(100),
        'Cred1': np.random.randn(100),
        'Normalised_FNT': np.random.randn(100),
        'geo_score': np.random.rand(100),
        'qsets_normalized_tat': np.random.rand(100),
        'instance_scores': np.random.rand(100),
        'lambda_wt': np.random.rand(100),
        'Target': np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    })

def test_feature_engineer_initialization():
    """Test FeatureEngineer initialization"""
    fe = FeatureEngineer()
    assert fe.scaler is not None
    assert isinstance(fe.numeric_features, list)

def test_identify_features(sample_data):
    """Test feature identification"""
    fe = FeatureEngineer()
    numeric_features, categorical_features = fe.identify_features(sample_data)
    
    assert isinstance(numeric_features, list)
    assert isinstance(categorical_features, list)
    assert 'Per1' in numeric_features
    assert 'Group' in categorical_features

def test_handle_missing_values(sample_data):
    """Test missing value handling"""
    # Introduce some missing values
    sample_data.loc[0:10, 'Per1'] = np.nan
    sample_data.loc[5:15, 'Group'] = np.nan
    
    fe = FeatureEngineer()
    fe.identify_features(sample_data)
    processed_data = fe.handle_missing_values(sample_data)
    
    assert processed_data['Per1'].isna().sum() == 0
    assert processed_data['Group'].isna().sum() == 0

def test_create_interaction_features(sample_data):
    """Test creation of interaction features"""
    fe = FeatureEngineer()
    processed_data = fe.create_interaction_features(sample_data)
    
    assert 'Per_ratio_1_2' in processed_data.columns
    assert 'Cred_mean' in processed_data.columns
    assert 'risk_score' in processed_data.columns

def test_scale_features(sample_data):
    """Test feature scaling"""
    fe = FeatureEngineer()
    fe.identify_features(sample_data)
    
    train_scaled, test_scaled = fe.scale_features(sample_data, sample_data.copy())
    
    # Check if scaling was applied
    assert not np.allclose(train_scaled['Per1'], sample_data['Per1'])
    assert train_scaled['Per1'].mean() < 1e-10  # Close to 0
    assert abs(train_scaled['Per1'].std() - 1.0) < 1e-10  # Close to 1 