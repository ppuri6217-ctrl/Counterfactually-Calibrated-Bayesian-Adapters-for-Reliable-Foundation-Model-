"""
Data processing and dataset handling for C²BA model.

This module contains classes for loading, preprocessing, and creating datasets
for the Counterfactually-Calibrated Bayesian Adapter model.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import os
import warnings

warnings.filterwarnings('ignore')


class HeartDiseaseDataset(Dataset):
    """Custom dataset for UCI Heart Disease with feature engineering"""
    
    def __init__(self, features, targets, transform=None):
        """
        Initialize the dataset.
        
        Args:
            features (np.ndarray): Feature matrix
            targets (np.ndarray): Target labels
            transform (callable, optional): Optional transform to be applied on features
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class DataProcessor:
    """Comprehensive data preprocessing pipeline"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_stats = {}
        self.embedding_dims = {
            'sex': 4, 'dataset': 6, 'cp': 6, 'restecg': 4, 'slope': 4, 'thal': 4
        }
        
    def engineer_features(self, df):
        """Advanced feature engineering"""
        df_eng = df.copy()
        
        # Polynomial interactions
        interactions = [
            ('age', 'trestbps'),
            ('age', 'thalch'), 
            ('chol', 'trestbps'),
            ('oldpeak', 'slope'),
            ('cp', 'exang')
        ]
        
        for feat1, feat2 in interactions:
            if feat1 in df_eng.columns and feat2 in df_eng.columns:
                df_eng[f'{feat1}_{feat2}_interaction'] = df_eng[feat1] * df_eng[feat2]
        
        # Missing value indicators
        for col in df_eng.columns:
            if df_eng[col].isna().sum() > len(df_eng) * 0.05:  # >5% missing
                df_eng[f'{col}_missing'] = df_eng[col].isna().astype(int)
        
        return df_eng
    
    def preprocess_data(self, df, is_training=True):
        """
        Complete preprocessing pipeline - adapted for real UCI heart data structure
        
        Args:
            df (pd.DataFrame): Input dataframe
            is_training (bool): Whether this is training data (affects fitting of scalers)
            
        Returns:
            tuple: (X_scaled, y) - preprocessed features and targets
        """
        df_processed = df.copy()
        
        # Handle common UCI heart dataset column names
        target_col = 'target' if 'target' in df.columns else 'num'
        
        # Standardize column names if needed
        column_mapping = {
            'num': 'target',  # UCI dataset sometimes uses 'num' for target
        }
        df_processed = df_processed.rename(columns=column_mapping)
        
        # Ensure target column exists
        if target_col not in df_processed.columns and 'target' not in df_processed.columns:
            if 'num' in df_processed.columns:
                df_processed['target'] = df_processed['num']
            else:
                raise ValueError("No target column found. Expected 'target' or 'num'")
        
        # For multi-class, convert to binary for simplicity in this demo
        if 'target' in df_processed.columns:
            max_target_value = df_processed['target'].max()
            if pd.isna(max_target_value):
                max_target_value = 0
            if max_target_value > 1:
                df_processed['target'] = (df_processed['target'] > 0).astype(int)
            
        # Feature engineering
        df_processed = self.engineer_features(df_processed)
        
        # Separate feature types
        feature_cols = [col for col in df_processed.columns if col != 'target']
        continuous_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
        binary_cols = ['sex', 'fbs', 'exang']
        categorical_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
        
        # Filter to existing columns
        continuous_cols = [col for col in continuous_cols if col in df_processed.columns]
        binary_cols = [col for col in binary_cols if col in df_processed.columns]
        categorical_cols = [col for col in categorical_cols if col in df_processed.columns]
        
        # Handle missing values
        for col in continuous_cols:
            if col in df_processed.columns:
                if is_training:
                    self.feature_stats[f'{col}_mean'] = df_processed[col].mean()
                fill_value = self.feature_stats.get(f'{col}_mean', df_processed[col].mean())
                df_processed[col].fillna(fill_value, inplace=True)
        
        for col in categorical_cols + binary_cols:
            if col in df_processed.columns:
                if is_training:
                    mode_val = df_processed[col].mode()
                    self.feature_stats[f'{col}_mode'] = mode_val[0] if len(mode_val) > 0 else 0
                fill_value = self.feature_stats.get(f'{col}_mode', 0)
                df_processed[col].fillna(fill_value, inplace=True)
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in df_processed.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    try:
                        df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        df_processed[col] = 0
        
        # Scale features
        feature_cols = [col for col in df_processed.columns if col != 'target']
        X = df_processed[feature_cols].values
        
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        y = df_processed['target'].values if 'target' in df_processed.columns else None
        
        return X_scaled, y


def load_heart_disease_data():
    """Load UCI Heart Disease dataset - adaptable to different input paths"""
    
    # Try multiple potential paths
    potential_paths = [
        'data/heart_disease_uci.csv',
        'data/heart.csv', 
        'data/uci-heart-disease/heart.csv',
        'heart.csv'  # If uploaded directly
    ]
    
    df = None
    for path in potential_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"✓ Successfully loaded data from: {path}")
                break
        except Exception as e:
            continue
    
    # If no dataset found, create synthetic data for demonstration
    if df is None:
        print("⚠ No heart disease dataset found in standard paths.")
        print("Creating synthetic dataset for demonstration...")
        df = create_synthetic_heart_data()
    
    return df


def create_synthetic_heart_data():
    """Create synthetic UCI Heart Disease-like dataset"""
    np.random.seed(42)
    n_samples = 1025  # Realistic size similar to UCI dataset
    
    # Create realistic synthetic data
    data = {
        'age': np.random.normal(54, 9, n_samples).clip(29, 77).astype(int),
        'sex': np.random.binomial(1, 0.68, n_samples),  # Male bias as in real data
        'cp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.16, 0.29, 0.08]),
        'trestbps': np.random.normal(131, 17, n_samples).clip(94, 200).astype(int),
        'chol': np.random.normal(246, 51, n_samples).clip(126, 564).astype(int),
        'fbs': np.random.binomial(1, 0.15, n_samples),
        'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.48, 0.48, 0.04]),
        'thalach': np.random.normal(149, 22, n_samples).clip(71, 202).astype(int),
        'exang': np.random.binomial(1, 0.33, n_samples),
        'oldpeak': np.random.exponential(1.04, n_samples).clip(0, 6.2).round(1),
        'slope': np.random.choice([0, 1, 2], n_samples, p=[0.21, 0.14, 0.65]),
        'ca': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.59, 0.21, 0.12, 0.06, 0.02]),
        'thal': np.random.choice([0, 1, 2, 3], n_samples, p=[0.02, 0.55, 0.36, 0.07])
    }
    
    # Create target variable with realistic medical correlations
    risk_factors = (
        0.02 * data['age'] +
        0.5 * (data['cp'] == 0) +  # Typical angina increases risk
        0.3 * data['exang'] +
        0.4 * (data['thal'] == 2) +  # Reversible defect
        0.2 * data['oldpeak'] +
        -0.01 * data['thalach'] +  # Higher heart rate = lower risk
        0.3 * (data['ca'] > 0) +  # Major vessels
        0.2 * data['sex'] +  # Male higher risk
        np.random.normal(0, 0.8, n_samples)
    )
    
    # Convert to binary (0=no disease, 1=disease) - more realistic for UCI data
    data['target'] = (risk_factors > np.percentile(risk_factors, 55)).astype(int)
    
    df = pd.DataFrame(data)
    
    # Add some missing values to make it realistic
    missing_cols = ['ca', 'thal']
    for col in missing_cols:
        missing_mask = np.random.random(len(df)) < 0.02  # 2% missing
        df.loc[missing_mask, col] = np.nan
    
    print(f"✓ Created synthetic dataset with {len(df)} samples")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df


def create_distribution_shifts(X, y, shift_type='age'):
    """
    Create different types of distribution shifts - adapted for binary classification
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        shift_type (str): Type of shift ('age', 'gender', 'severity', 'feature')
        
    Returns:
        tuple: (train_mask, test_mask) - boolean masks for train/test split
    """
    
    if shift_type == 'age':
        # Age-based split (assuming age is first column after scaling)
        age_median = np.median(X[:, 0])  # Age is typically first feature
        train_mask = X[:, 0] <= age_median  # Younger patients for training
        test_mask = X[:, 0] > age_median    # Older patients for testing
    
    elif shift_type == 'gender':
        # Gender-based split (assuming sex is in features)
        sex_col = None
        for i in range(min(10, X.shape[1])):  # Check first 10 features
            unique_vals = np.unique(X[:, i])
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                sex_col = i
                break
        
        if sex_col is not None:
            # Create gender imbalance
            male_indices = np.where(X[:, sex_col] > 0.5)[0]
            female_indices = np.where(X[:, sex_col] <= 0.5)[0]
            
            # Train: 70% male, Test: balanced
            n_train = len(X) // 2
            if len(male_indices) > 0 and len(female_indices) > 0:
                train_male_count = min(int(0.7 * n_train), len(male_indices))
                train_female_count = min(n_train - train_male_count, len(female_indices))
                
                train_indices = np.concatenate([
                    np.random.choice(male_indices, train_male_count, replace=False),
                    np.random.choice(female_indices, train_female_count, replace=False)
                ])
                
                train_mask = np.zeros(len(X), dtype=bool)
                train_mask[train_indices] = True
                test_mask = ~train_mask
            else:
                # Fallback to random split
                train_mask = np.random.random(len(X)) < 0.6
                test_mask = ~train_mask
        else:
            # Fallback to random split
            train_mask = np.random.random(len(X)) < 0.6
            test_mask = ~train_mask
    
    elif shift_type == 'severity':
        # Severity-based split for binary classification
        positive_indices = np.where(y == 1)[0]
        negative_indices = np.where(y == 0)[0]
        
        if len(positive_indices) > 0 and len(negative_indices) > 0:
            # Train: 80% negative cases (mild), Test: balanced
            n_train = len(X) // 2
            train_neg_count = min(int(0.8 * n_train), len(negative_indices))
            train_pos_count = min(n_train - train_neg_count, len(positive_indices))
            
            train_indices = np.concatenate([
                np.random.choice(negative_indices, train_neg_count, replace=False),
                np.random.choice(positive_indices, train_pos_count, replace=False)
            ])
            
            train_mask = np.zeros(len(X), dtype=bool)
            train_mask[train_indices] = True
            test_mask = ~train_mask
        else:
            # Fallback to random split
            train_mask = np.random.random(len(X)) < 0.6
            test_mask = ~train_mask
    
    elif shift_type == 'feature':
        # Feature-based shift: split based on a continuous feature
        feature_col = min(2, X.shape[1] - 1)  # Use 3rd feature
        feature_median = np.median(X[:, feature_col])
        train_mask = X[:, feature_col] <= feature_median
        test_mask = X[:, feature_col] > feature_median
    
    else:
        # Random split as baseline
        train_mask = np.random.random(len(X)) < 0.6
        test_mask = ~train_mask
    
    # Ensure both sets have both classes for binary classification
    train_classes = np.unique(y[train_mask])
    test_classes = np.unique(y[test_mask])
    
    if len(train_classes) < 2 or len(test_classes) < 2:
        print("⚠ Warning: Unbalanced class distribution detected. Using stratified split.")
        # Fallback to stratified split
        train_indices, test_indices = train_test_split(
            np.arange(len(X)), test_size=0.4, random_state=42, stratify=y
        )
        train_mask = np.zeros(len(X), dtype=bool)
        test_mask = np.zeros(len(X), dtype=bool)
        train_mask[train_indices] = True
        test_mask[test_indices] = True
    
    return train_mask, test_mask
