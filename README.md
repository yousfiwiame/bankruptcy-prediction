# Corporate Bankruptcy Prediction System

This project implements both traditional machine learning and deep learning approaches to predict corporate bankruptcies using financial data. The system is designed to forecast whether a company will go bankrupt in the following year based on historical financial indicators.

## Table of Contents 

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Machine Learning Approach](#machine-learning-approach)
- [Deep Learning Approach](#deep-learning-approach)
- [Model Comparison](#model-comparison)
- [Explainability Analysis](#explainability-analysis)
- [Usage](#usage)
- [Requirements](#requirements)

## Overview 

Corporate bankruptcy prediction is an essential task in financial risk analysis. This project:

- Processes historical financial data from American companies
- Implements various machine learning and deep learning models
- Evaluates model performance with an emphasis on correctly identifying bankruptcy risks
- Provides time-series analysis for sequential financial data
- Addresses class imbalance through resampling techniques
- Employs model explainability to identify key bankruptcy indicators

## Dataset

The project uses the "american_bankruptcy.csv" dataset which contains:

- Financial data from multiple companies across several years
- Various financial indicators (features prefixed with 'X')
- Binary classification: 'failed' (bankrupt) or 'alive' (operational)
- Temporal information allowing for time-series prediction

Key dataset characteristics:

- Significant class imbalance (few bankruptcy cases compared to operational cases)
- Temporal dependencies in financial indicators
- Structured by company name and year

## Project Structure

The project is organized into two main components:

1. Machine Learning Models 
   - Traditional classification algorithms
   - Feature preprocessing and scaling
   - Model optimization via cross-validation
   - Class imbalance handling

2. Deep Learning Models 
   - Sequential data processing (2-year and 5-year lookback periods)
   - Various neural network architectures (LSTM, GRU, CNN, BiLSTM)
   - Optimization techniques and callbacks
   - Performance visualization and analysis

## Machine Learning Approach

**Data Preprocessing**

- Target engineering: Creation of 'bankrupt_next_year' label
- Feature standardization
- Train-test splitting with stratification
- SMOTE oversampling for class balance

**Models Implemented**

1. Random Forest
   - Ensemble-based approach with class weighting
   - Hyperparameter tuning via GridSearchCV

2. Logistic Regression
   - Linear classifier with balanced class weights
   - L2 regularization

3. Multi-Layer Perceptron (MLP)
   - Neural network with configurable hidden layers
   - Early stopping to prevent overfitting

4. XGBoost
   - Gradient boosting implementation
   - Optimized learning rate and tree depth

**Pipeline Integration**

- Comprehensive preprocessing-model pipelines
- Cross-validation for robust evaluation
- Metrics focus: ROC-AUC, F1-score, precision, recall

## Deep Learning Approach

**Sequential Data Processing**

The project employs two different sequence lengths:

- Short-term (2-year lookback)
- Long-term (5-year lookback)

**Architecture Implementation**

1. LSTM (Long Short-Term Memory)
   - Basic and optimized variants
   - Dropout regularization
   - Return sequences for stacked architecture

2. GRU (Gated Recurrent Unit)
   - Memory-efficient recurrent architecture
   - Comparable performance to LSTM with fewer parameters

3. 1D CNN (Convolutional Neural Network)
   - Temporal convolutions
   - GlobalMaxPooling for feature extraction

4. BiLSTM (Bidirectional LSTM)
   - Processes sequences in both directions
   - Captures forward and backward dependencies
   - Best overall performance in the time-series context

**Optimization Techniques**

- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Class weighting for imbalanced data
- Dropout regularization
- Batch normalization

## Model Comparison

**Performance Metrics**

- *Accuracy* : Overall correctness
- *Precision* : Percentage of correctly identified bankruptcies
- *Recall* : Ability to detect actual bankruptcies
- *F1-Score* : Harmonic mean of precision and recall
- *ROC-AUC* : Area under the ROC curve

**Key Findings**

- BiLSTM models generally outperformed other architectures
- Class balancing significantly improved minority class detection
- 5-year sequences provided better predictive power than 2-year sequences
- Traditional ML models performed well with proper preprocessing

## Explainability Analysis

The project integrates SHAP (SHapley Additive exPlanations) to interpret model decisions:

- Feature importance extraction from BiLSTM embeddings
- XGBoost used as an interpretable proxy model
- Visualization of key financial indicators driving bankruptcy prediction

## Usage

**Model Training**

```python

<pre lang="markdown"> ```python # Machine Learning Models from sklearn.ensemble import RandomForestClassifier from sklearn.preprocessing import StandardScaler from imblearn.pipeline import Pipeline as ImbPipeline from imblearn.over_sampling import SMOTE # Create ML pipeline pipeline = ImbPipeline(steps=[ ("scaler", StandardScaler()), ("smote", SMOTE(random_state=42)), ("model", RandomForestClassifier()) ]) # Fit model pipeline.fit(X_train, y_train) # Deep Learning Models from tensorflow.keras.models import Sequential from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional # Create BiLSTM model bilstm_model = Sequential([ Bidirectional(LSTM(64, return_sequences=True), input_shape=(seq_len, n_features)), Dropout(0.3), Bidirectional(LSTM(32)), Dropout(0.3), Dense(32, activation='relu'), Dropout(0.3), Dense(1, activation='sigmoid') ]) # Compile model bilstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Train model history = bilstm_model.fit( X_train, y_train, epochs=70, batch_size=64, validation_split=0.2, class_weight=class_weights, callbacks=[early_stop, lr_reduce] ) ``` </pre>


## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- TensorFlow 2.x
- Keras
- XGBoost
- SHAP
- matplotlib
- seaborn
