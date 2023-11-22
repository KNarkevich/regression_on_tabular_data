import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from config import FEATURES


def train_model(data_path, model_save_path):
    df = pd.read_csv(data_path)
    
    X = df[FEATURES]
    y = df['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([('scaler', StandardScaler()), 
                    ('lr', RandomForestRegressor())])
    
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f'Validation RMSE: {rmse}')

    joblib.dump(pipe, model_save_path)


if __name__ == "__main__":
    data_path = 'data/train.csv'
    model_save_path = 'model/model.joblib'
    train_model(data_path, model_save_path)
