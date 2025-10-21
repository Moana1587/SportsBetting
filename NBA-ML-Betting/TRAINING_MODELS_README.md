# NBA Betting Model Training Guide

This guide explains how to train machine learning models for NBA betting predictions, including Moneyline (ML), Over/Under (OU), and Spread predictions.

## Overview

The training suite includes multiple model types for different prediction tasks:

### Classification Models
- **Moneyline (ML) Classification**: Predicts which team wins (Home vs Away)
- **Over/Under (OU) Classification**: Predicts if total points will be Over, Under, or Push

### Regression Models
- **Spread Regression**: Predicts the point spread value
- **ML Difference Regression**: Predicts the difference between home and away ML odds

## Model Types

### 1. XGBoost Models
- **XGBoost_Model_ML.py**: Original ML classification model
- **XGBoost_Model_UO.py**: Original Over/Under classification model
- **XGBoost_Model_Spread.py**: NEW - Spread regression model
- **XGBoost_Model_ML_Classification.py**: NEW - ML classification with odds features
- **XGBoost_Model_ML_Regression.py**: NEW - ML difference regression

### 2. Neural Network Models
- **NN_Model_ML.py**: Original ML classification neural network
- **NN_Model_UO.py**: Original Over/Under classification neural network
- **NN_Model_Spread.py**: NEW - Spread regression neural network
- **NN_Model_ML_Classification.py**: NEW - ML classification neural network with odds

## Quick Start

### Run All Models
```bash
python train_all_models.py
```

### Run Individual Models
```bash
cd src/Train-Models
python XGBoost_Model_Spread.py
python XGBoost_Model_ML_Classification.py
```

### Test a Model
```bash
python test_spread_model.py
```

## Model Performance

### Spread Regression Model
- **R² Score**: ~0.527 (52.7% variance explained)
- **MAE**: ~2.26 points
- **MSE**: ~9.51
- **Features**: 106 team statistics features

### ML Classification Model
- **Accuracy**: Typically 60-70%
- **Features**: Team stats + ML odds
- **Target**: Binary classification (Home win vs Away win)

## Data Requirements

The models require the following columns in the dataset:
- Team statistics (106 features)
- `Spread`: Point spread values
- `ML_Home`, `ML_Away`: Moneyline odds
- `Home-Team-Win`: Binary target (1 = Home wins, 0 = Away wins)
- `OU`, `OU-Cover`: Over/Under line and result

## Model Outputs

### Spread Model
- Predicts actual spread values
- Higher values = Home team favored more
- Lower values = Away team favored more

### ML Classification Model
- Predicts probability of home team winning
- Output: 0-1 probability score
- >0.5 = Home team favored
- <0.5 = Away team favored

## Training Parameters

### XGBoost Models
- **Max Depth**: 3-6
- **Learning Rate**: 0.01
- **Epochs**: 750-1000
- **Objective**: 
  - Classification: `binary:logistic` or `multi:softprob`
  - Regression: `reg:squarederror`

### Neural Network Models
- **Architecture**: 128 → 64 → 32 → 1
- **Activation**: ReLU (hidden), Sigmoid/Linear (output)
- **Dropout**: 0.2-0.3
- **Epochs**: 100
- **Batch Size**: 32

## Model Files

Trained models are saved in the `Models/` directory:
- `XGBoost_Spread_Value_*.json`: Spread regression models
- `XGBoost_*%_ML_Classification.json`: ML classification models
- `NN_Spread_Model_*`: Neural network spread models
- `NN_ML_Classification_Model_*`: Neural network ML models

## Usage in Predictions

The trained models can be used in the prediction pipeline:

```python
# Load spread model
xgb_spread = xgb.Booster()
xgb_spread.load_model('Models/XGBoost_Spread_Value_9.510_MSE.json')

# Make predictions
spread_predictions = xgb_spread.predict(xgb.DMatrix(features))
```

## Troubleshooting

### Common Issues
1. **Unicode Errors**: Fixed by removing emoji characters
2. **XGBoost Version**: Ensure compatible XGBoost version
3. **Memory Issues**: Reduce batch size or number of iterations
4. **Data Format**: Ensure all features are numeric

### Performance Tips
1. **Feature Engineering**: Add more relevant team statistics
2. **Hyperparameter Tuning**: Experiment with different parameters
3. **Cross-Validation**: Use k-fold validation for better estimates
4. **Ensemble Methods**: Combine multiple models for better predictions

## Next Steps

1. **Feature Engineering**: Add more advanced statistics
2. **Hyperparameter Optimization**: Use grid search or Bayesian optimization
3. **Model Ensembling**: Combine XGBoost and Neural Network predictions
4. **Real-time Updates**: Retrain models with new data
5. **Backtesting**: Validate models on historical data

## Dependencies

Required packages:
- `xgboost`
- `tensorflow`
- `scikit-learn`
- `pandas`
- `numpy`
- `tqdm`
- `sqlite3`
