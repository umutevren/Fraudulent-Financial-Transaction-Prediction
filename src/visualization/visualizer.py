import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve, average_precision_score
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudVisualizer:
    def __init__(self, output_dir: str = 'output/plots'):
        """
        Initialize the visualizer with an output directory for saving plots.
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for all plots
        plt.style.use('seaborn-v0_8')
        sns.set_theme()
        
        # Set custom color palette for better visualization
        self.colors = sns.color_palette("husl", 8)
        self.fraud_colors = ['#2ecc71', '#e74c3c']  # Green for normal, Red for fraud
    
    def set_figure_style(self):
        """Set consistent style for all figures"""
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def plot_feature_distributions(self, df: pd.DataFrame, features: List[str], 
                                 target_col: Optional[str] = None):
        """
        Plot distribution of features, optionally split by target class.
        
        Args:
            df: DataFrame containing features
            features: List of feature names to plot
            target_col: Optional target column for class-wise distribution
        """
        self.set_figure_style()
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle('Feature Distributions by Class', fontsize=16, y=1.02)
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            if target_col:
                # Plot with enhanced styling
                sns.kdeplot(data=df, x=feature, hue=target_col, ax=axes[idx],
                          palette=self.fraud_colors, common_norm=False)
                axes[idx].set_title(f'Distribution of {feature}', pad=10)
                axes[idx].legend(title='Class', labels=['Normal', 'Fraud'])
            else:
                sns.histplot(data=df, x=feature, ax=axes[idx], color=self.colors[0])
            axes[idx].set_title(f'Distribution of {feature}')
        
        # Remove empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Feature distributions plot saved")
    
    def plot_correlation_matrix(self, df: pd.DataFrame, features: List[str]):
        """
        Plot correlation matrix for selected features with enhanced styling.
        
        Args:
            df: DataFrame containing features
            features: List of feature names to include
        """
        self.set_figure_style()
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[features].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, fmt='.2f', square=True,
                   annot_kws={'size': 8})
        
        plt.title('Feature Correlation Matrix', pad=20, fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Correlation matrix plot saved")
    
    def plot_class_distribution(self, y: pd.Series):
        """
        Plot the distribution of target classes with enhanced styling.
        
        Args:
            y: Target series
        """
        self.set_figure_style()
        plt.figure(figsize=(10, 6))
        class_counts = y.value_counts()
        
        # Create bar plot with custom colors
        ax = sns.barplot(x=['Normal', 'Fraud'], y=class_counts.values, 
                        palette=self.fraud_colors)
        
        plt.title('Class Distribution in Dataset', pad=20, fontsize=14)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Add count and percentage labels
        total = len(y)
        for i, count in enumerate(class_counts.values):
            percentage = count / total * 100
            ax.text(i, count, f'Count: {count:,}\n({percentage:.1f}%)', 
                   ha='center', va='bottom')
        
        # Add grid for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Class distribution plot saved")
    
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_scores: np.ndarray,
                              title: str = 'Feature Importance'):
        """
        Plot feature importance scores with enhanced styling.
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            title: Plot title
        """
        self.set_figure_style()
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1]
        top_n = 20
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(top_n), 
                       importance_scores[indices][:top_n],
                       color=self.colors[0],
                       alpha=0.8)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, i, f'{width:.3f}', 
                    va='center', ha='left', fontsize=10)
        
        plt.yticks(range(top_n), [feature_names[i] for i in indices][:top_n])
        plt.title(title, pad=20, fontsize=14)
        plt.xlabel('Importance Score', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Feature importance plot saved")
    
    def plot_roc_curves(self, y_true: np.ndarray, 
                       model_probas: Dict[str, np.ndarray]):
        """
        Plot ROC curves for multiple models with enhanced styling.
        
        Args:
            y_true: True target values
            model_probas: Dictionary of model names and their prediction probabilities
        """
        self.set_figure_style()
        plt.figure(figsize=(10, 8))
        
        for idx, (model_name, y_pred_proba) in enumerate(model_probas.items()):
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = average_precision_score(y_true, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})',
                    color=self.colors[idx], linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        plt.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', pad=20, fontsize=14)
        plt.legend(loc='lower right', frameon=True, framealpha=0.8)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("ROC curves plot saved")
    
    def plot_precision_recall_curves(self, y_true: np.ndarray,
                                   model_probas: Dict[str, np.ndarray]):
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            y_true: True target values
            model_probas: Dictionary of model names and their prediction probabilities
        """
        self.set_figure_style()
        plt.figure(figsize=(10, 8))
        
        for idx, (model_name, y_pred_proba) in enumerate(model_probas.items()):
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            
            plt.plot(recall, precision, 
                    label=f'{model_name} (AP = {avg_precision:.3f})',
                    color=self.colors[idx], linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', pad=20, fontsize=14)
        plt.legend(loc='lower left', frameon=True, framealpha=0.8)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Precision-Recall curves plot saved")
    
    def plot_confusion_matrices(self, y_true: np.ndarray, 
                              model_predictions: Dict[str, np.ndarray]):
        """
        Plot confusion matrices for multiple models with enhanced styling.
        
        Args:
            y_true: True target values
            model_predictions: Dictionary of model names and their predictions
        """
        self.set_figure_style()
        n_models = len(model_predictions)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 7*n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (model_name, y_pred) in enumerate(model_predictions.items()):
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate percentages
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot both raw counts and percentages
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=axes[idx])
            axes[idx].set_title(f'Confusion Matrix - {model_name}', pad=20)
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('True', fontsize=10)
            
            # Add percentage labels
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[idx].text(j+0.5, i+0.1, f'({cm_normalized[i,j]*100:.1f}%)',
                                 ha='center', va='center')
        
        # Remove empty subplots
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Confusion matrices plot saved")
    
    def plot_metric_comparison(self, metrics: Dict[str, Dict[str, float]]):
        """
        Plot comparison of different metrics across models with enhanced styling.
        
        Args:
            metrics: Dictionary of model names and their performance metrics
        """
        self.set_figure_style()
        models = list(metrics.keys())
        metric_names = list(metrics[models[0]].keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Metrics Comparison', fontsize=16, y=1.02)
        axes = axes.flatten()
        
        for idx, metric in enumerate(metric_names):
            metric_values = [metrics[model][metric] for model in models]
            
            bars = sns.barplot(x=models, y=metric_values, ax=axes[idx],
                             palette=self.colors)
            
            axes[idx].set_title(f'{metric.upper()}', pad=10)
            axes[idx].set_ylim(0, 1)
            axes[idx].set_xlabel('Model')
            axes[idx].set_ylabel('Score')
            
            # Add value labels
            for i, v in enumerate(metric_values):
                axes[idx].text(i, v, f'{v:.3f}', ha='center', va='bottom')
            
            # Rotate x-labels for better readability
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metric_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Metric comparison plot saved") 