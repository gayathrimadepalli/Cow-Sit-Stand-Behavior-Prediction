# Cow Sit/Stand Behavior Prediction

## Project Overview
This project uses machine learning to predict whether cows are sitting or standing based on sensor data collected from devices attached to the animals. The model analyzes patterns in accelerometer, gyroscope, and magnetometer readings to classify cow posture accurately.

## Problem Statement
Monitoring cow behavior is crucial for animal welfare and farm management. This project automates the detection of cow posture (sitting vs. standing) using sensor data, which can help farmers:
- Monitor animal health and comfort
- Detect abnormal behaviors early
- Optimize farm management strategies
- Reduce manual observation time

## Dataset Description
The project uses two datasets:
- `data.csv`: Training data with labeled sit/stand behavior (300 records)
- `evaluation_ext.csv`: Evaluation data for predictions (30 records)

Each record contains:
- Device ID
- Timestamp (cdate)
- Sensor readings in array format:
  - Accelerometer data (ax, ay, az)
  - Gyroscope data (gx, gy, gz)
  - Magnetometer data (mx, my, mz)
- Target label: sit_stand (for training data only)

## Methodology

### Data Preprocessing
- Parsing array-like strings into numerical data
- Converting timestamps into structured datetime features
- Handling missing values
- Feature standardization

### Feature Engineering
For each sensor array, we extracted six statistical features:
- Mean (average reading)
- Standard deviation (variation in readings)
- Minimum value
- Maximum value
- Median value
- Range (max-min)

Additionally, time-based features were extracted:
- Hour of day
- Day of month
- Month
- Day of week

### Model Selection
Primary model: **Random Forest Classifier**
- Handles high-dimensional data well
- Resistant to overfitting
- Provides feature importance rankings
- Works well with both numerical and categorical features

### Model Pipeline
1. Data loading and preprocessing
2. Feature extraction from sensor arrays
3. Train-test split (80%-20%)
4. Feature standardization
5. Model training with hyperparameter tuning using GridSearchCV
6. Evaluation on validation set
7. Prediction on evaluation data

## Results

### Model Performance
- Validation accuracy: 81.67%
- Sitting class precision: 53%, recall: 67%
- Standing class precision: 91%, recall: 85%

### Feature Importance
Top features in prediction:
1. Unnamed: 0 (row identifier)
2. Hour of day
3. Day of week
4. Day of month

### Distribution of Predictions
- Standing: 93.3%
- Sitting: 6.7%

## Insights
- Cows exhibit regular patterns in sitting/standing behavior throughout the day
- Temporal features (hour, day) proved highly influential for prediction
- Statistical features from sensors (especially means and standard deviations) capture important behavioral patterns
- The model demonstrates good precision for standing behavior but is less precise for sitting behavior

## Future Improvements
- Apply more advanced time-series feature extraction
- Implement deep learning approaches for raw sensor data
- Explore frequency-domain features (Fourier transforms)
- Use sliding window techniques for sequential behavior analysis
- Implement SHAP values or similar tools for better model interpretability
- Collect more balanced data (especially for sitting behavior)

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage
1. Ensure you have the required datasets: `data.csv` and `evaluation_ext.csv`
2. Run the main script to train the model and generate predictions
3. Predictions will be saved to `evaluation_with_predictions.csv`

## Challenges
- Parsing complex string representations of sensor arrays
- Balancing feature richness with dimensionality
- Managing class imbalance (more standing than sitting instances)
- Handling potential sensor noise and data quality issues
