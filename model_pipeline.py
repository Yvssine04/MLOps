import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


def load_data(data_path="Churn_Modelling.csv"):
    data = pd.read_csv(data_path)
    print(f"Data loaded from {data_path}: {data.shape[0]} lignes")
    return data


def clean_data(data):
    data = data.drop(["Surname", "Geography"], axis=1)
    data = data.dropna()
    print(f"Data cleaned: {data.shape[0]} lignes, {data.shape[1]} colonnes")
    return data


def encode_data(data):
    encoder = LabelEncoder()
    data["Gender"] = encoder.fit_transform(data["Gender"])
    print("Data encoded: Gender -> 0/1")
    return data


def split_data(data):
    X = data.drop(["Exited", "RowNumber", "CustomerId"], axis=1)
    y = data["Exited"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    print(f"Data split: {len(x_train)} train / {len(x_test)} test")
    return x_train, x_test, y_train, y_test


def scale_data(x_train, x_test, scaler_path="scaler.joblib"):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, scaler_path)
    print(f"Data scaled, scaler saved to {scaler_path}")
    return x_train_scaled, x_test_scaled


def prepare_data(data_path="Churn_Modelling.csv"):
    data = load_data(data_path)
    data = clean_data(data)
    data = encode_data(data)
    x_train, x_test, y_train, y_test = split_data(data)
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
    print("Data prepared.")
    return x_train_scaled, x_test_scaled, y_train, y_test


def train_model(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(x_train, y_train)
    print(f"Model trained on {len(x_train)} lignes")
    return model


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return accuracy


def save_model(model, model_path="model.joblib"):
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path="model.joblib"):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def predict(features, model_path="model.joblib"):
    model = load_model(model_path)
    predictions = model.predict(features)
    print(f"Predictions done: {len(predictions)} lignes")
    return predictions
