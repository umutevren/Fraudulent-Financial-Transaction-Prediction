import logging
from pathlib import Path
from data.data_loader import prepare_datasets
from features.feature_engineering import FeatureEngineer
from models.model_trainer import ModelTrainer
from visualization.visualizer import FraudVisualizer
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the fraud detection pipeline.
    """
    try:
        # Initialize visualizer
        visualizer = FraudVisualizer()
        
        # Step 1: Load and merge datasets
        logger.info("Loading and merging datasets...")
        train_df, test_df = prepare_datasets()
        
        # Save target column before feature engineering
        y = train_df['Target'].copy()
        
        # Visualize initial data distributions
        logger.info("Generating initial data visualizations...")
        numeric_features = (
            [f"Per{i}" for i in range(1, 10)] +
            [f"Dem{i}" for i in range(1, 10)] +
            [f"Cred{i}" for i in range(1, 7)] +
            ['Normalised_FNT', 'geo_score', 'qsets_normalized_tat', 
             'instance_scores', 'lambda_wt']
        )
        
        # Plot initial data visualizations
        logger.info("Plotting data distributions and correlations...")
        visualizer.plot_feature_distributions(train_df, numeric_features[:9], 'Target')
        visualizer.plot_correlation_matrix(train_df, numeric_features)
        visualizer.plot_class_distribution(train_df['Target'])
        
        # Step 2: Feature engineering
        logger.info("Performing feature engineering...")
        feature_engineer = FeatureEngineer()
        X_train, X_test = feature_engineer.prepare_features(train_df, test_df)
        
        # Visualize engineered features
        logger.info("Plotting engineered feature distributions...")
        engineered_features = [
            'Per_ratio_1_2', 'Per_ratio_3_4', 'Cred_mean',
            'Dem_mean', 'Per_mean', 'risk_score'
        ]
        visualizer.plot_feature_distributions(
            pd.concat([X_train, pd.Series(y, name='Target')], axis=1),
            engineered_features,
            'Target'
        )
        
        # Step 3: Model training and evaluation
        logger.info("Training and evaluating models...")
        model_trainer = ModelTrainer()
        
        # Prepare training data with SMOTE
        X_train_resampled, X_val, y_train_resampled, y_val = model_trainer.prepare_training_data(
            X_train, y
        )
        
        # Visualize class distribution after SMOTE
        logger.info("Plotting class distribution after SMOTE...")
        visualizer.plot_class_distribution(pd.Series(y_train_resampled))
        
        # Train and evaluate models
        results = model_trainer.train_and_evaluate(
            X_train_resampled, X_val, y_train_resampled, y_val
        )
        
        # Collect predictions and probabilities for visualization
        model_predictions = {}
        model_probas = {}
        for name, model in model_trainer.models.items():
            model_predictions[name] = model.predict(X_val)
            model_probas[name] = model.predict_proba(X_val)[:, 1]
        
        # Generate comprehensive model performance visualizations
        logger.info("Generating model performance visualizations...")
        
        # Plot ROC and Precision-Recall curves
        visualizer.plot_roc_curves(y_val, model_probas)
        visualizer.plot_precision_recall_curves(y_val, model_probas)
        
        # Plot confusion matrices
        visualizer.plot_confusion_matrices(y_val, model_predictions)
        
        # Plot metric comparison
        visualizer.plot_metric_comparison(results)
        
        # Plot feature importance for the best model if it's a tree-based model
        if hasattr(model_trainer.best_model, 'feature_importances_'):
            visualizer.plot_feature_importance(
                list(X_train.columns),
                model_trainer.best_model.feature_importances_,
                f'Feature Importance - {model_trainer.best_model_name}'
            )
        
        # Step 4: Generate predictions for test set
        logger.info("Generating predictions for test set...")
        test_predictions = model_trainer.predict(X_test)
        test_probabilities = model_trainer.predict_proba(X_test)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'id': test_df['id'],
            'prediction': test_predictions,
            'probability': test_probabilities
        })
        
        # Step 5: Save results
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        # Save predictions
        submission_path = output_dir / 'predictions.csv'
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Predictions saved to {submission_path}")
        
        # Save best model
        model_trainer.save_model(output_dir='models')
        
        logger.info("Fraud detection pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in fraud detection pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 