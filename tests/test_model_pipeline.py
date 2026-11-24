import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_pipeline import (  # noqa: E402
    clean_data,
    encode_data,
    evaluate_model,
    load_data,
    load_model,
    predict,
    prepare_data,
    save_model,
    scale_data,
    split_data,
    train_model,
)


def test_load_data():
    data = load_data()
    assert data.shape[0] > 0
    assert "Exited" in data.columns


def test_clean_data():
    data = clean_data(load_data())
    assert "Surname" not in data.columns
    assert "Geography" not in data.columns
    assert data.isna().sum().sum() == 0


def test_encode_data():
    data = encode_data(clean_data(load_data()))
    assert set(data["Gender"].unique()) <= {0, 1}


def test_split_data():
    data = encode_data(clean_data(load_data()))
    x_train, x_test, y_train, y_test = split_data(data)
    assert len(x_train) == int(len(data) * 0.8)
    assert "Exited" not in x_train.columns
    assert len(x_train) == len(y_train)


def test_scale_data(tmp_path):
    data = encode_data(clean_data(load_data()))
    x_train, x_test, _, _ = split_data(data)
    x_train_s, x_test_s = scale_data(x_train, x_test, tmp_path / "scaler.joblib")
    assert abs(x_train_s.mean()) < 0.01
    assert (tmp_path / "scaler.joblib").exists()


def test_prepare_data():
    x_train, x_test, y_train, y_test = prepare_data()
    assert len(x_train) + len(x_test) == 10000
    assert len(x_test) == len(y_test)


def test_train_and_evaluate_model():
    x_train, x_test, y_train, y_test = prepare_data()
    model = train_model(x_train, y_train)
    accuracy = evaluate_model(model, x_test, y_test)
    assert 0.8 <= accuracy <= 1.0


def test_save_and_load_model(tmp_path):
    x_train, x_test, y_train, y_test = prepare_data()
    model = train_model(x_train[:500], y_train[:500])
    path = tmp_path / "model.joblib"
    save_model(model, path)
    loaded = load_model(path)
    assert np.array_equal(model.predict(x_test), loaded.predict(x_test))


def test_predict(tmp_path):
    x_train, x_test, y_train, y_test = prepare_data()
    model = train_model(x_train[:500], y_train[:500])
    path = tmp_path / "model.joblib"
    save_model(model, path)
    predictions = predict(x_test, path)
    assert len(predictions) == len(x_test)
    assert set(np.unique(predictions)) <= {0, 1}
