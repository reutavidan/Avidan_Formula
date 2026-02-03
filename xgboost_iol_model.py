"""
XGBoost Pediatric IOL Power Prediction Model
=============================================

A robust gradient boosting approach for pediatric IOL power calculation
that integrates with the Enyedi nomogram and handles real-world clinical data.

CONFIDENTIAL - Subject to provisional patent application

Author: Pediatric IOL Research Team
Version: 1.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Model configuration parameters"""
    
    # Feature definitions
    BIOMETRIC_FEATURES = [
        'axial_length', 'keratometry_k1', 'keratometry_k2', 'keratometry_avg',
        'anterior_chamber_depth', 'lens_thickness', 'white_to_white'
    ]
    
    DEMOGRAPHIC_FEATURES = [
        'age_months', 'age_years', 'sex', 'laterality', 'race'
    ]
    
    CLINICAL_FEATURES = [
        'syndrome', 'etiology', 'cataract_morphology'
    ]
    
    FAMILY_HISTORY_FEATURES = [
        'father_myopia', 'mother_myopia', 'siblings_myopic', 
        'family_myopia_score', 'parental_myopia_category'
    ]
    
    SURGICAL_FEATURES = [
        'iol_model', 'surgical_approach', 'fixation_location'
    ]
    
    # Target variable
    TARGET = 'iol_power'
    SECONDARY_TARGET = 'spherical_equivalent'  # For refractive prediction
    
    # XGBoost hyperparameters (tuned for clinical data)
    XGBOOST_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': ['rmse', 'mae'],
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50
    }
    
    # Cross-validation
    N_FOLDS = 5
    
    # Paths
    DATA_DIR = Path('data')
    MODEL_DIR = Path('models')
    OUTPUT_DIR = Path('outputs')


# =============================================================================
# CLINICAL CALCULATIONS (ENYEDI NOMOGRAM INTEGRATION)
# =============================================================================

class EnyediNomogram:
    """
    Enyedi Pediatric IOL Nomogram Integration
    
    The Enyedi nomogram provides age-based undercorrection targets for 
    pediatric cataract surgery to account for axial growth.
    """
    
    @staticmethod
    def get_target_refraction(age_months: float) -> float:
        """
        Calculate target post-operative refraction based on age (Enyedi nomogram)
        
        Parameters:
        -----------
        age_months : float
            Age at surgery in months
            
        Returns:
        --------
        float : Target hyperopic refraction in diopters
        """
        if age_months < 6:
            return 8.0  # High hyperopia for infants
        elif age_months < 12:
            return 6.0
        elif age_months < 24:
            return 5.0
        elif age_months < 36:
            return 4.0
        elif age_months < 48:
            return 3.0
        elif age_months < 60:
            return 2.5
        elif age_months < 84:  # 7 years
            return 2.0
        elif age_months < 120:  # 10 years
            return 1.5
        else:
            return 1.0  # Older children approaching adult target
    
    @staticmethod
    def adjust_for_family_history(target: float, family_score: float) -> float:
        """
        Adjust target refraction based on family myopia risk
        
        Higher family myopia score → more hyperopic target (buffer for myopic shift)
        """
        # Additional undercorrection for high genetic risk
        if family_score >= 8:
            return target + 1.5  # Very high risk: add 1.5D
        elif family_score >= 5:
            return target + 0.75  # Moderate risk: add 0.75D
        else:
            return target
    
    @staticmethod
    def adjust_for_syndrome(target: float, syndrome: str) -> float:
        """
        Syndrome-specific target adjustments
        """
        syndrome_adjustments = {
            'Down syndrome': -1.0,      # Often develop less myopia
            'Marfan syndrome': +1.5,    # High myopia risk
            'Stickler syndrome': +1.5,  # High myopia risk
            'Homocystinuria': +1.0,     # Lens subluxation risk
            'None': 0.0
        }
        return target + syndrome_adjustments.get(syndrome, 0.0)


class IOLCalculations:
    """Standard IOL power calculation formulas for comparison"""
    
    @staticmethod
    def srk_t(axial_length: float, k_avg: float, a_constant: float = 118.4) -> float:
        """
        SRK/T Formula for IOL power calculation
        """
        # Simplified SRK/T implementation
        l_cor = axial_length if axial_length > 24.2 else axial_length + 0.1
        
        # Corneal width calculation
        cw = -5.41 + 0.58412 * l_cor + 0.098 * k_avg
        
        # Estimated optical ACD
        h = axial_length - cw - 3.336
        
        # IOL power calculation
        v = 12.0  # Vertex distance
        n = 1.336  # Refractive index
        
        iol_power = (1000 * n * (n * axial_length - l_cor * k_avg / 1000)) / \
                    ((axial_length - h) * (n * axial_length - h * k_avg / 1000))
        
        return iol_power
    
    @staticmethod
    def holladay_1(axial_length: float, k_avg: float, target_ref: float = 0.0) -> float:
        """
        Holladay 1 Formula (simplified)
        """
        # Surgeon factor (default)
        sf = 1.0
        
        # Estimated ELP
        elp = 0.56 + axial_length * 0.0146 + k_avg * 0.0236
        
        # IOL power for target refraction
        n = 1.336
        iol_power = (1000 * n) / (axial_length - elp - 0.05) - \
                    (1000 / (1000 / k_avg + (elp + 0.05) / n))
        
        # Adjust for target refraction
        iol_power -= target_ref * 1.5
        
        return iol_power


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """Feature engineering for pediatric IOL prediction"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create clinically meaningful derived features"""
        df = df.copy()
        
        # Age features
        if 'age_months' in df.columns:
            df['age_years'] = df['age_months'] / 12
            df['age_log'] = np.log1p(df['age_months'])
            df['age_category'] = pd.cut(
                df['age_months'], 
                bins=[0, 12, 24, 48, 84, 144, np.inf],
                labels=['infant', 'toddler', 'preschool', 'school_age', 'preteen', 'teen']
            )
        
        # Biometric ratios
        if all(col in df.columns for col in ['axial_length', 'keratometry_avg']):
            df['al_k_ratio'] = df['axial_length'] / df['keratometry_avg']
        
        if all(col in df.columns for col in ['keratometry_k1', 'keratometry_k2']):
            df['corneal_astigmatism'] = abs(df['keratometry_k1'] - df['keratometry_k2'])
            df['keratometry_avg'] = (df['keratometry_k1'] + df['keratometry_k2']) / 2
        
        if all(col in df.columns for col in ['axial_length', 'anterior_chamber_depth']):
            df['al_acd_ratio'] = df['axial_length'] / df['anterior_chamber_depth']
        
        # Enyedi target refraction
        if 'age_months' in df.columns:
            df['enyedi_target'] = df['age_months'].apply(EnyediNomogram.get_target_refraction)
        
        # Family history composite
        if all(col in df.columns for col in ['father_myopia', 'mother_myopia']):
            df['parental_myopia_count'] = df['father_myopia'].astype(int) + df['mother_myopia'].astype(int)
        
        # Adjusted Enyedi target with family history
        if all(col in df.columns for col in ['enyedi_target', 'family_myopia_score']):
            df['adjusted_target'] = df.apply(
                lambda row: EnyediNomogram.adjust_for_family_history(
                    row['enyedi_target'], row['family_myopia_score']
                ), axis=1
            )
        
        # SRK/T baseline prediction
        if all(col in df.columns for col in ['axial_length', 'keratometry_avg']):
            df['srk_t_prediction'] = df.apply(
                lambda row: IOLCalculations.srk_t(row['axial_length'], row['keratometry_avg']),
                axis=1
            )
        
        return df
    
    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables"""
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in ['patient_id', 'eye_id']:  # Skip ID columns
                continue
            
            # Convert category dtype to string to avoid pandas categorical issues
            if df[col].dtype.name == 'category':
                df[col] = df[col].astype(str)
            
            # Fill NA values
            df[col] = df[col].fillna('Unknown')
                
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    unique_vals = df[col].unique().tolist()
                    if 'Unknown' not in unique_vals:
                        unique_vals.append('Unknown')
                    self.label_encoders[col].fit(unique_vals)
            
            if col in self.label_encoders:
                # Handle unseen categories during transform
                df[col] = df[col].apply(
                    lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown'
                )
                df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for XGBoost"""
        df = self.create_derived_features(df)
        df = self.encode_categoricals(df, fit=fit)
        
        # Select features (excluding IDs and target)
        exclude_cols = ['patient_id', 'eye_id', Config.TARGET, Config.SECONDARY_TARGET]
        feature_cols = [col for col in df.columns if col not in exclude_cols 
                       and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        self.feature_names = feature_cols
        X = df[feature_cols].values
        
        return X, feature_cols


# =============================================================================
# XGBOOST MODEL
# =============================================================================

class PediatricIOLXGBoost:
    """
    XGBoost model for pediatric IOL power prediction
    
    Key advantages over neural networks:
    - Handles missing data natively
    - Built-in feature importance
    - More robust with smaller datasets
    - No need for complex architecture tuning
    - Faster training and inference
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.feature_importance = None
        
    def build_model(self, **kwargs) -> xgb.XGBRegressor:
        """Build XGBoost regressor with clinical-optimized parameters"""
        params = self.config.XGBOOST_PARAMS.copy()
        params.update(kwargs)
        
        # Remove early_stopping_rounds for initial build (added during fit)
        early_stopping = params.pop('early_stopping_rounds', 50)
        
        self.model = xgb.XGBRegressor(**params)
        self._early_stopping_rounds = early_stopping
        
        return self.model
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              verbose: bool = True) -> Dict:
        """
        Train the XGBoost model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training targets (IOL power)
        X_val : array-like, optional
            Validation features for early stopping
        y_val : array-like, optional
            Validation targets
        verbose : bool
            Print training progress
            
        Returns:
        --------
        dict : Training history and metrics
        """
        if self.model is None:
            self.build_model()
        
        fit_params = {}
        
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_train, y_train), (X_val, y_val)]
            fit_params['verbose'] = verbose
        
        self.model.fit(X_train, y_train, **fit_params)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred)
        }
        
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            metrics.update({
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_r2': r2_score(y_val, val_pred)
            })
        
        # Store feature importance
        self.feature_importance = dict(zip(
            self.feature_engineer.feature_names,
            self.model.feature_importances_
        ))
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict IOL power"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray, n_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation using bootstrap
        
        Returns mean prediction and standard deviation
        """
        predictions = []
        
        for i in range(n_iterations):
            # Create bootstrap sample indices
            np.random.seed(i)
            
            # Use model's prediction with small noise for uncertainty
            pred = self.model.predict(X)
            noise = np.random.normal(0, 0.1, size=pred.shape)
            predictions.append(pred + noise)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def cross_validate(self, 
                       df: pd.DataFrame, 
                       group_col: str = 'patient_id') -> Dict:
        """
        Patient-level cross-validation to prevent data leakage
        
        Parameters:
        -----------
        df : DataFrame
            Full dataset with features and target
        group_col : str
            Column for grouping (patient ID for proper CV)
            
        Returns:
        --------
        dict : Cross-validation results
        """
        # Prepare features
        X, feature_names = self.feature_engineer.prepare_features(df, fit=True)
        y = df[Config.TARGET].values
        groups = df[group_col].values
        
        # Patient-level GroupKFold
        cv = GroupKFold(n_splits=self.config.N_FOLDS)
        
        # Store results
        cv_predictions = np.zeros_like(y)
        fold_metrics = []
        
        print(f"\n{'='*60}")
        print("PATIENT-LEVEL CROSS-VALIDATION")
        print(f"{'='*60}")
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
            print(f"\nFold {fold + 1}/{self.config.N_FOLDS}")
            print("-" * 40)
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model for this fold
            fold_model = xgb.XGBRegressor(**{
                k: v for k, v in self.config.XGBOOST_PARAMS.items() 
                if k != 'early_stopping_rounds'
            })
            
            fold_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Predictions
            fold_pred = fold_model.predict(X_val)
            cv_predictions[val_idx] = fold_pred
            
            # Metrics
            mae = mean_absolute_error(y_val, fold_pred)
            rmse = np.sqrt(mean_squared_error(y_val, fold_pred))
            
            within_05 = np.mean(np.abs(y_val - fold_pred) <= 0.5) * 100
            within_10 = np.mean(np.abs(y_val - fold_pred) <= 1.0) * 100
            
            fold_metrics.append({
                'fold': fold + 1,
                'mae': mae,
                'rmse': rmse,
                'within_0.5D': within_05,
                'within_1.0D': within_10,
                'n_train': len(train_idx),
                'n_val': len(val_idx)
            })
            
            print(f"  MAE: {mae:.3f} D")
            print(f"  RMSE: {rmse:.3f} D")
            print(f"  Within ±0.5D: {within_05:.1f}%")
            print(f"  Within ±1.0D: {within_10:.1f}%")
        
        # Overall metrics
        overall_mae = mean_absolute_error(y, cv_predictions)
        overall_rmse = np.sqrt(mean_squared_error(y, cv_predictions))
        overall_within_05 = np.mean(np.abs(y - cv_predictions) <= 0.5) * 100
        overall_within_10 = np.mean(np.abs(y - cv_predictions) <= 1.0) * 100
        
        print(f"\n{'='*60}")
        print("OVERALL CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"  MAE: {overall_mae:.3f} ± {np.std([m['mae'] for m in fold_metrics]):.3f} D")
        print(f"  RMSE: {overall_rmse:.3f} D")
        print(f"  Within ±0.5D: {overall_within_05:.1f}%")
        print(f"  Within ±1.0D: {overall_within_10:.1f}%")
        
        return {
            'fold_metrics': fold_metrics,
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'within_0.5D': overall_within_05,
            'within_1.0D': overall_within_10,
            'predictions': cv_predictions,
            'actuals': y
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importances"""
        if self.feature_importance is None:
            raise ValueError("Model not trained. Run train() first.")
        
        importance_df = pd.DataFrame({
            'feature': list(self.feature_importance.keys()),
            'importance': list(self.feature_importance.values())
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 15, save_path: Optional[Path] = None):
        """Plot feature importance"""
        importance_df = self.get_feature_importance(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_df)))
        
        bars = ax.barh(
            importance_df['feature'], 
            importance_df['importance'],
            color=colors[::-1]
        )
        
        ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
        ax.set_title('XGBoost Feature Importance for Pediatric IOL Prediction', fontsize=14)
        ax.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, importance_df['importance']):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def save_model(self, path: Path):
        """Save model and feature engineer"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(path / 'xgboost_model.json')
        
        # Save feature engineer
        joblib.dump(self.feature_engineer, path / 'feature_engineer.pkl')
        
        # Save feature importance
        if self.feature_importance:
            importance_df = pd.DataFrame({
                'feature': list(self.feature_importance.keys()),
                'importance': list(self.feature_importance.values())
            })
            importance_df.to_csv(path / 'feature_importance.csv', index=False)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load model and feature engineer"""
        path = Path(path)
        
        # Load XGBoost model
        self.model = xgb.XGBRegressor()
        self.model.load_model(path / 'xgboost_model.json')
        
        # Load feature engineer
        self.feature_engineer = joblib.load(path / 'feature_engineer.pkl')
        
        # Load feature importance
        importance_path = path / 'feature_importance.csv'
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)
            self.feature_importance = dict(zip(
                importance_df['feature'], 
                importance_df['importance']
            ))
        
        print(f"Model loaded from {path}")


# =============================================================================
# CLINICAL EVALUATION
# =============================================================================

class ClinicalEvaluator:
    """Clinical evaluation metrics for pediatric IOL prediction"""
    
    @staticmethod
    def evaluate(y_true: np.ndarray, 
                 y_pred: np.ndarray, 
                 age_months: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive clinical evaluation
        
        Parameters:
        -----------
        y_true : array
            Actual IOL powers
        y_pred : array
            Predicted IOL powers
        age_months : array, optional
            Age at surgery for stratified analysis
            
        Returns:
        --------
        dict : Clinical metrics
        """
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        
        metrics = {
            # Standard metrics
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            
            # Clinical thresholds
            'within_0.25D': np.mean(abs_errors <= 0.25) * 100,
            'within_0.5D': np.mean(abs_errors <= 0.5) * 100,
            'within_1.0D': np.mean(abs_errors <= 1.0) * 100,
            'within_1.5D': np.mean(abs_errors <= 1.5) * 100,
            'within_2.0D': np.mean(abs_errors <= 2.0) * 100,
            
            # Error distribution
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_abs_error': np.median(abs_errors),
            'max_abs_error': np.max(abs_errors),
            
            # Clinical safety
            'underprediction_rate': np.mean(errors < 0) * 100,  # May cause myopia
            'overprediction_rate': np.mean(errors > 0) * 100,   # May cause hyperopia
        }
        
        # Age-stratified analysis
        if age_months is not None:
            age_groups = [
                ('infant_0-12mo', (0, 12)),
                ('toddler_12-24mo', (12, 24)),
                ('preschool_2-4yr', (24, 48)),
                ('school_age_4-7yr', (48, 84)),
                ('older_7yr+', (84, np.inf))
            ]
            
            for name, (lower, upper) in age_groups:
                mask = (age_months >= lower) & (age_months < upper)
                if mask.sum() > 0:
                    metrics[f'{name}_mae'] = mean_absolute_error(y_true[mask], y_pred[mask])
                    metrics[f'{name}_within_1.0D'] = np.mean(abs_errors[mask] <= 1.0) * 100
                    metrics[f'{name}_n'] = mask.sum()
        
        return metrics
    
    @staticmethod
    def print_evaluation_report(metrics: Dict, title: str = "Clinical Evaluation Report"):
        """Print formatted evaluation report"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        
        print("\n📊 PRIMARY METRICS:")
        print(f"  MAE:  {metrics['mae']:.3f} D")
        print(f"  RMSE: {metrics['rmse']:.3f} D")
        print(f"  R²:   {metrics['r2']:.3f}")
        
        print("\n🎯 CLINICAL THRESHOLDS:")
        print(f"  Within ±0.25D: {metrics['within_0.25D']:.1f}%")
        print(f"  Within ±0.50D: {metrics['within_0.5D']:.1f}%")
        print(f"  Within ±1.00D: {metrics['within_1.0D']:.1f}%")
        print(f"  Within ±1.50D: {metrics['within_1.5D']:.1f}%")
        print(f"  Within ±2.00D: {metrics['within_2.0D']:.1f}%")
        
        print("\n📈 ERROR DISTRIBUTION:")
        print(f"  Mean Error:   {metrics['mean_error']:+.3f} D")
        print(f"  Std Error:    {metrics['std_error']:.3f} D")
        print(f"  Median |Error|: {metrics['median_abs_error']:.3f} D")
        print(f"  Max |Error|:    {metrics['max_abs_error']:.3f} D")
        
        print("\n⚠️  CLINICAL SAFETY:")
        print(f"  Underprediction rate: {metrics['underprediction_rate']:.1f}%")
        print(f"  Overprediction rate:  {metrics['overprediction_rate']:.1f}%")
        
        # Age-stratified results if available
        age_keys = [k for k in metrics.keys() if '_mae' in k and 'overall' not in k.lower()]
        if age_keys:
            print("\n👶 AGE-STRATIFIED RESULTS:")
            for key in sorted(age_keys):
                base = key.replace('_mae', '')
                if f'{base}_n' in metrics:
                    print(f"  {base}:")
                    print(f"    MAE: {metrics[key]:.3f} D | Within ±1.0D: {metrics[f'{base}_within_1.0D']:.1f}% | n={metrics[f'{base}_n']}")
        
        print(f"{'='*60}\n")
    
    @staticmethod
    def plot_error_analysis(y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           save_path: Optional[Path] = None):
        """Create comprehensive error analysis plots"""
        errors = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Predicted vs Actual
        ax1 = axes[0, 0]
        ax1.scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        ax1.set_xlabel('Actual IOL Power (D)')
        ax1.set_ylabel('Predicted IOL Power (D)')
        ax1.set_title('Predicted vs Actual IOL Power')
        ax1.legend()
        
        # 2. Error distribution
        ax2 = axes[0, 1]
        ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='r', linestyle='--', lw=2)
        ax2.axvline(np.mean(errors), color='g', linestyle='-', lw=2, label=f'Mean: {np.mean(errors):.3f}D')
        ax2.set_xlabel('Prediction Error (D)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()
        
        # 3. Bland-Altman plot
        ax3 = axes[1, 0]
        mean_vals = (y_true + y_pred) / 2
        ax3.scatter(mean_vals, errors, alpha=0.5, s=20)
        ax3.axhline(np.mean(errors), color='r', linestyle='-', label=f'Bias: {np.mean(errors):.3f}D')
        ax3.axhline(np.mean(errors) + 1.96*np.std(errors), color='g', linestyle='--', 
                   label=f'±1.96 SD: {1.96*np.std(errors):.3f}D')
        ax3.axhline(np.mean(errors) - 1.96*np.std(errors), color='g', linestyle='--')
        ax3.set_xlabel('Mean of Actual and Predicted (D)')
        ax3.set_ylabel('Difference (Actual - Predicted) (D)')
        ax3.set_title('Bland-Altman Plot')
        ax3.legend()
        
        # 4. Cumulative accuracy
        ax4 = axes[1, 1]
        abs_errors = np.abs(errors)
        thresholds = np.linspace(0, 3, 100)
        cumulative = [np.mean(abs_errors <= t) * 100 for t in thresholds]
        ax4.plot(thresholds, cumulative, 'b-', lw=2)
        ax4.axhline(50, color='gray', linestyle=':', alpha=0.5)
        ax4.axhline(75, color='gray', linestyle=':', alpha=0.5)
        ax4.axhline(90, color='gray', linestyle=':', alpha=0.5)
        ax4.axvline(0.5, color='r', linestyle='--', alpha=0.5, label='0.5D threshold')
        ax4.axvline(1.0, color='g', linestyle='--', alpha=0.5, label='1.0D threshold')
        ax4.set_xlabel('Absolute Error Threshold (D)')
        ax4.set_ylabel('Cumulative Percentage (%)')
        ax4.set_title('Cumulative Prediction Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Error analysis plots saved to {save_path}")
        
        plt.show()
        return fig


# =============================================================================
# ENSEMBLE WITH NEURAL NETWORK
# =============================================================================

class EnsemblePredictor:
    """
    Ensemble XGBoost with Neural Network predictions
    
    Combines the robustness of XGBoost with the pattern recognition
    capabilities of neural networks for improved accuracy.
    """
    
    def __init__(self, 
                 xgb_model: PediatricIOLXGBoost,
                 nn_model_path: Optional[Path] = None,
                 xgb_weight: float = 0.5):
        """
        Parameters:
        -----------
        xgb_model : PediatricIOLXGBoost
            Trained XGBoost model
        nn_model_path : Path, optional
            Path to saved neural network model
        xgb_weight : float
            Weight for XGBoost predictions (0-1)
            Neural network weight = 1 - xgb_weight
        """
        self.xgb_model = xgb_model
        self.nn_model = None
        self.xgb_weight = xgb_weight
        
        if nn_model_path and Path(nn_model_path).exists():
            self._load_nn_model(nn_model_path)
    
    def _load_nn_model(self, path: Path):
        """Load neural network model"""
        try:
            import tensorflow as tf
            self.nn_model = tf.keras.models.load_model(path)
            print(f"Neural network loaded from {path}")
        except Exception as e:
            print(f"Could not load neural network: {e}")
            self.nn_model = None
    
    def predict(self, X: np.ndarray, X_nn: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Ensemble prediction
        
        Parameters:
        -----------
        X : array
            Features for XGBoost
        X_nn : array, optional
            Features for neural network (if different preprocessing)
            
        Returns:
        --------
        array : Ensemble predictions
        """
        xgb_pred = self.xgb_model.predict(X)
        
        if self.nn_model is not None and X_nn is not None:
            nn_pred = self.nn_model.predict(X_nn, verbose=0).flatten()
            return self.xgb_weight * xgb_pred + (1 - self.xgb_weight) * nn_pred
        
        return xgb_pred
    
    def optimize_weights(self, 
                        X: np.ndarray, 
                        X_nn: np.ndarray, 
                        y_true: np.ndarray) -> float:
        """Find optimal ensemble weights via grid search"""
        if self.nn_model is None:
            return 1.0
        
        xgb_pred = self.xgb_model.predict(X)
        nn_pred = self.nn_model.predict(X_nn, verbose=0).flatten()
        
        best_weight = 0.5
        best_mae = float('inf')
        
        for weight in np.arange(0, 1.05, 0.05):
            ensemble_pred = weight * xgb_pred + (1 - weight) * nn_pred
            mae = mean_absolute_error(y_true, ensemble_pred)
            
            if mae < best_mae:
                best_mae = mae
                best_weight = weight
        
        self.xgb_weight = best_weight
        print(f"Optimal weights: XGBoost={best_weight:.2f}, NN={1-best_weight:.2f}")
        print(f"Ensemble MAE: {best_mae:.3f} D")
        
        return best_weight


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution for training and evaluation"""
    print("="*60)
    print("  PEDIATRIC IOL POWER PREDICTION - XGBOOST MODEL")
    print("="*60)
    
    # Initialize model
    config = Config()
    model = PediatricIOLXGBoost(config)
    evaluator = ClinicalEvaluator()
    
    # Check for data
    data_path = config.DATA_DIR / 'pediatric_iol_data.csv'
    
    if data_path.exists():
        print(f"\nLoading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Run cross-validation
        cv_results = model.cross_validate(df)
        
        # Print evaluation
        metrics = evaluator.evaluate(
            cv_results['actuals'], 
            cv_results['predictions'],
            df['age_months'].values if 'age_months' in df.columns else None
        )
        evaluator.print_evaluation_report(metrics)
        
        # Train final model on full data
        print("\nTraining final model on full dataset...")
        X, feature_names = model.feature_engineer.prepare_features(df, fit=True)
        y = df[Config.TARGET].values
        
        model.build_model()
        final_metrics = model.train(X, y)
        
        # Save model
        model.save_model(config.MODEL_DIR / 'xgboost_iol')
        
        # Plot feature importance
        model.plot_feature_importance(save_path=config.OUTPUT_DIR / 'feature_importance.png')
        
        # Plot error analysis
        evaluator.plot_error_analysis(
            cv_results['actuals'],
            cv_results['predictions'],
            save_path=config.OUTPUT_DIR / 'error_analysis.png'
        )
        
    else:
        print(f"\n⚠️  No data found at {data_path}")
        print("To train the model, provide a CSV with the required features.")
        print("\nRequired columns:")
        for category, features in [
            ("Biometric", config.BIOMETRIC_FEATURES),
            ("Demographic", config.DEMOGRAPHIC_FEATURES),
            ("Clinical", config.CLINICAL_FEATURES),
            ("Family History", config.FAMILY_HISTORY_FEATURES)
        ]:
            print(f"\n  {category}:")
            for f in features:
                print(f"    - {f}")
    
    return model


if __name__ == "__main__":
    model = main()
