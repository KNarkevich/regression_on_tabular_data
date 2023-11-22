# Regression on Tabular Data

This project focuses on building a machine learning model for regression on tabular data. The dataset, `train.csv`, comprises 53 anonymized features and a target column. The objective is to develop a model capable of predicting the target based on the provided features. The target metric for evaluation is RMSE (Root Mean Square Error).

## Repository Structure

1. **Jupyter Notebook (exploratory_data_analysis.ipynb):**
   - [Jupyter Notebook](exploratory_data_analysis.ipynb) containing Exploratory Data Analysis (EDA) on the dataset.

2. **Python Script for Model Training (train.py):**
   - [train.py](train.py) - Python script for model training. Handles loading the dataset, preprocessing, model training, and saving the trained model.

3. **Python Script for Model Inference (predict.py):**
   - [predict.py](predict.py) - Python script for model inference on the test data (`hidden_test.csv`). Loads the trained model and generates predictions.

4. **Prediction Results File (predictions.csv):**
   - [predictions.csv](predictions.csv) - File containing predictions on the hidden test data.

## Usage

Call the `predict` function to make prediction:

```python
from predict import predict 

test_data_path = 'data/hidden_test.csv'
output_path = 'predictions.csv'
predict(test_data_path, output_path)

```

## Generating the Random Forest Model

To create and train the Random Forest model, execute the `train.py` script. This script contains the necessary steps for generating the model without the need for additional code. Simply run the following command in your terminal:

```bash
python3 train.py
```


