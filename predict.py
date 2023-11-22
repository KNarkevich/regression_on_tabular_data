import pandas as pd
import joblib
from config import FEATURES
from config import model_path

def predict(data_path, output_path):
    
    test_data = pd.read_csv(data_path)

    model = joblib.load(model_path)

    predictions = model.predict(test_data[FEATURES])
    
    result_df = pd.DataFrame({'Prediction': predictions})
    result_df.to_csv(output_path, index=False)
    print(f'Predictions saved to {output_path}')


if __name__ == "__main__":
    test_data_path = 'data/hidden_test.csv'
    output_path = 'predictions.csv'
    predict(test_data_path, output_path)
